"""ChatGPT data export: `<export>/conversations-*.json`, each a JSON array.

Four facts about this format decide the mapping.

**A conversation is a tree, not a transcript.** `mapping` holds every node the
conversation ever had — regenerated answers, edited prompts, abandoned
descendants — related by `parent`/`children`. `current_node` records the leaf
the reader last saw, so walking parents from it reproduces the visible thread
exactly. Object order and timestamps do not declare the active branch. Using
either would emit alternate branches as sequential turns and inflate every
count that follows.

**The export is text plus media, and only the text is read.** Image, audio, and
real-time audio/video components are counted and dropped. `audio_transcription`
parts are the export's own transcript of a voice input; they are treated as
audio, not as prose, so nothing this adapter emits was ever spoken rather than
typed. No decoder, no recognizer, no description stands in for what was
dropped.

**Reasoning prose is present here.** A `thoughts` block carries readable
summary text, so `chars.thinking` is measured and `chars.signature` stays zero
— in contrast to the CLI format `claude`, where the thinking block is empty
beside an encrypted signature.

**Usage and working directory are not recorded.** Every `usage` key is zero for
this whole adapter and `cwd` is always empty — see `records.TOKEN_KEYS`: a
whole-adapter zero means "not recorded", never "none happened".

Provenance departs from the line-oriented formats in one documented way. A
shard is a pretty-printed array whose elements run to hundreds of kilobytes, so
a message has no source line. `provenance.line` is the message's position in the
active-branch traversal, which is what orders a session; `provenance.turn_id` is
the node id, which locates it exactly; and `provenance.sha256` is over the
message node's canonical serialization, reproducible from the export by anyone
holding it:

    sha256(records.dumps(mapping[turn_id]["message"]))
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..records import Turn, blank_turn, dumps
from .base import (
    SchemaDrift,
    SessionRef,
    Span,
    array_spans,
    bounded,
    read_span,
    sorted_counts,
)

name = "chatgpt"

#: This format has no working-directory concept. Without the declaration
#: `scan.sessions.meta` cannot take its fast path and enumeration costs a full
#: parse of the corpus before a single turn is measured.
records_cwd = False

#: `author.role` -> canonical role. `system` and `tool` are events: harness and
#: tool traffic, not either side of the conversation.
ROLES = {"user": "user", "assistant": "assistant", "system": "event", "tool": "event"}

#: Content types that can carry something the operator typed. A user record of
#: any other type is machine-authored, whatever channel it arrived on.
TYPED_TYPES = {"text", "multimodal_text"}

#: Media components inside a `multimodal_text` block. None of them yields
#: canonical text. `audio_transcription` is listed as audio deliberately: the
#: record identifies it as a transcription of a voice input, which makes it
#: audio-derived rather than written.
IMAGE_PARTS = {"image_asset_pointer"}
AUDIO_PARTS = {
    "audio_asset_pointer",
    "audio_transcription",
    "real_time_user_audio_video_asset_pointer",
}

#: Content types whose body is a tool result rather than conversation text.
TOOL_RESULT_TYPES = {"execution_output", "tether_browsing_display", "tether_quote"}

#: How long a vendor discriminator may be before it is truncated on its way into
#: a diagnostic. A category name is schema; an unbounded string from an unknown
#: record shape is not, and diagnostics are printed.
MAX_DISCRIMINATOR = 40


class _Diagnostics:
    """Content-neutral aggregates for one scan.

    Category names and closed-vocabulary discriminators only. No title, body,
    identifier, filename, or address reaches this structure, which is what makes
    it safe to print from a run over a private corpus.
    """

    __slots__ = ("counts", "content_types", "skipped", "errors", "unknown_types")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.counts: Counter = Counter()
        self.content_types: Counter = Counter()
        self.skipped: Counter = Counter()
        self.errors: Counter = Counter()
        self.unknown_types: Counter = Counter()


_DIAG = _Diagnostics()


def diagnostics() -> dict[str, dict[str, int]]:
    """What the last scan skipped, failed on, and could not resolve.

    Sorted on the way out, so two runs over one corpus print the same bytes.
    """
    return {
        "counts": sorted_counts(_DIAG.counts),
        "content_types": sorted_counts(_DIAG.content_types),
        "skipped_content": sorted_counts(_DIAG.skipped),
        "errors": sorted_counts(_DIAG.errors),
        "unknown_types": sorted_counts(_DIAG.unknown_types),
    }


def _tag(value: object) -> str:
    text = str(value or "")
    return (text[:MAX_DISCRIMINATOR] or "<absent>")


# --------------------------------------------------------------------------
# Locating conversations
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Span:
    """One conversation object, located in its shard but not yet read."""

    path: str
    index: int              # position in the shard's array
    start: int              # byte offset of the object
    end: int                # byte offset one past it
    conversation_id: str
    create_time: str
    empty: bool = False      # a keyless object: nothing drifted, nothing readable


#: Per-shard spans, and the winning span per conversation id. Both are derived
#: from file contents alone, so a cached lookup answers what a re-read would.
_INDEX: dict[str, list[_Span]] = {}
_SPANS: dict[str, _Span] = {}


def default_root() -> Path:
    return Path.home() / "corpus" / "chatgpt"


def _shards(root: Path) -> list[Path]:
    """Shards under an export directory, or under a directory of exports."""
    return sorted({*root.glob("conversations*.json"), *root.glob("*/conversations*.json")})


def _scan_shard(path: Path) -> list[_Span]:
    """Every conversation in one shard, located by byte span and identified.

    The span scan itself is `base.array_spans`; this adds the two fields that
    make a conversation addressable — its own id, and the creation time that
    buckets it into a window without parsing a message.
    """
    out: list[_Span] = []
    for span, conv in array_spans(path):
        cid = ""
        created = ""
        empty = isinstance(conv, dict) and not conv
        if isinstance(conv, dict):
            cid = str(conv.get("conversation_id") or conv.get("id") or "")
            created = _iso(conv.get("create_time"))
        out.append(_Span(span.path, span.index, span.start, span.end,
                         cid or f"{path.stem}:{span.index}", created, empty))
    return out


def _shard_index(path: Path) -> list[_Span]:
    key = str(path)
    if key not in _INDEX:
        _INDEX[key] = _scan_shard(path)
    return _INDEX[key]


def sessions(root: Path | None = None, include_subagents: bool = True) -> list[SessionRef]:
    """One session per conversation, across every shard under `root`.

    `root` is an export directory or a directory of them. A conversation id
    repeated across exports is kept once — the first in shard order, which is
    sorted — and the repeat is counted rather than emitted twice. Identity comes
    from the export's own `conversation_id`; no text is compared, because two
    conversations that read alike are not the same conversation.
    """
    root = Path(root) if root else default_root()
    _DIAG.reset()
    _SPANS.clear()
    if not root.is_dir():
        return []

    shards = _shards(root)
    _DIAG.counts["files_scanned"] = len(shards)
    for shard in shards:
        for span in _shard_index(shard):
            _DIAG.counts["conversations_discovered"] += 1
            # An object with no keys at all is not a reshape of the format:
            # nothing drifted, there is simply nothing there. Skip it before
            # it can reach `probe`, where a keyless object would read as drift
            # and take every readable conversation in the root down with it.
            if span.empty:
                _DIAG.counts["empty_conversation_objects"] += 1
                continue
            if span.conversation_id in _SPANS:
                _DIAG.counts["duplicate_conversation_ids"] += 1
                continue
            _SPANS[span.conversation_id] = span

    return sorted(
        (SessionRef(name, cid, span.path, project=Path(span.path).parent.name)
         for cid, span in _SPANS.items()),
        key=lambda ref: ref.session_id,
    )


def _span_for(ref: SessionRef) -> _Span | None:
    """The located conversation, whether or not `sessions()` ran in this process."""
    span = _SPANS.get(ref.session_id)
    if span is not None and span.path == ref.path:
        return span
    for candidate in _shard_index(Path(ref.path)):
        if candidate.conversation_id == ref.session_id:
            return candidate
    return None


def _conversation(span: _Span) -> dict:
    obj = read_span(Span(span.path, span.index, span.start, span.end))
    return obj if isinstance(obj, dict) else {}


def first_ts(ref: SessionRef) -> str:
    """The conversation's own creation time, read from the index.

    Sessions bucket by their first recorded timestamp, and this format records
    one per conversation, so no message needs parsing to place a session.
    """
    span = _span_for(ref)
    return span.create_time if span else ""


def probe(ref: SessionRef) -> None:
    """Raise `SchemaDrift` when a conversation carries no readable node tree."""
    span = _span_for(ref)
    if span is None:
        raise SchemaDrift(f"{ref.path}: no conversation {ref.session_id!r}")
    conv = _conversation(span)
    mapping = conv.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise SchemaDrift(
            f"{ref.path}: conversation carries no `mapping`; keys are {sorted(conv)[:6]}"
        )
    if not any(isinstance(node, dict) and "parent" in node
               for node in mapping.values()):
        raise SchemaDrift(
            f"{ref.path}: no node carries `parent`; "
            "the branch relation this adapter reconstructs is gone"
        )


# --------------------------------------------------------------------------
# Branch reconstruction
# --------------------------------------------------------------------------


def _active_branch(conv: dict, mapping: dict) -> tuple[list[str], str]:
    """The exported visible thread, root first, and how it was established.

    `current_node` is the export's own record of what the reader last saw.
    Without it the branch is still determined when the tree has a single leaf —
    there is then only one path to take — and is otherwise reported as
    ambiguous rather than flattened. No timing heuristic decides it.
    """
    node = conv.get("current_node")
    how = "current_node"
    if not (isinstance(node, str) and node in mapping):
        parents = {
            entry.get("parent") for entry in mapping.values()
            if isinstance(entry, dict) and isinstance(entry.get("parent"), str)
        }
        leaves = sorted(
            nid for nid, entry in mapping.items()
            if isinstance(entry, dict) and (
                not entry.get("children") if "children" in entry else nid not in parents
            )
        )
        if len(leaves) != 1:
            return [], "ambiguous"
        node, how = leaves[0], "sole_leaf"

    path: list[str] = []
    walked: set[str] = set()
    while isinstance(node, str) and node not in walked:
        walked.add(node)
        entry = mapping.get(node)
        if not isinstance(entry, dict):
            _DIAG.errors["invalid_parent_reference"] += 1
            break
        path.append(node)
        node = entry.get("parent")
    path.reverse()
    return path, how


def turns(ref: SessionRef) -> Iterator[Turn]:
    """The active branch of one conversation, root first, as canonical turns."""
    span = _span_for(ref)
    if span is None:
        _DIAG.errors["malformed_conversation"] += 1
        return
    conv = _conversation(span)
    mapping = conv.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        _DIAG.errors["malformed_conversation"] += 1
        return

    branch, how = _active_branch(conv, mapping)
    if how == "ambiguous":
        _DIAG.errors["ambiguous_active_branch"] += 1
        _DIAG.counts["nodes_total"] += len(mapping)
        return

    _DIAG.counts["conversations_emitted"] += 1
    _DIAG.counts["nodes_total"] += len(mapping)
    _DIAG.counts[f"branch_resolved_by_{how}"] += 1
    _DIAG.counts["active_branch_nodes"] += len(branch)
    _DIAG.counts["alternate_branch_nodes"] += len(mapping) - len(branch)

    message_ids: set[str] = set()
    for position, node_id in enumerate(branch):
        node = mapping[node_id]
        message = node.get("message")
        if not isinstance(message, dict):
            if node.get("parent") is None:
                _DIAG.counts["root_nodes"] += 1
            else:
                _DIAG.errors["missing_message_body"] += 1
            continue
        mid = str(message.get("id") or "")
        if mid and mid in message_ids:
            _DIAG.counts["duplicate_message_ids"] += 1
        message_ids.add(mid)
        yield _turn(ref, span, position, node_id, message)


# --------------------------------------------------------------------------
# One message -> one Turn
# --------------------------------------------------------------------------


def _turn(ref: SessionRef, span: _Span, position: int, node_id: str,
          message: dict) -> Turn:
    author = message.get("author")
    author = author if isinstance(author, dict) else {}
    source_role = str(author.get("role") or "")
    role = ROLES.get(source_role)
    if role is None:
        _DIAG.errors["unsupported_role"] += 1
        _DIAG.unknown_types[_tag(source_role)] += 1
        role = "event"

    content = message.get("content")
    content = content if isinstance(content, dict) else {}
    ctype = str(content.get("content_type") or "")
    _DIAG.content_types[_tag(ctype)] += 1

    metadata = message.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    t = blank_turn(name, span.path, position, _digest(message), ref.session_id,
                   node_id, role, ctype or source_role)
    t["ts"] = _iso(message.get("create_time"))
    t["model"] = str(metadata.get("model_slug") or "")
    t["effort"] = str(metadata.get("thinking_effort") or "")

    body = _content(ctype, content, t)
    _tools(message, author, source_role, role, t)

    finish = metadata.get("finish_details")
    finish_type = str(finish.get("type") or "") if isinstance(finish, dict) else ""
    t["flags"]["interrupted"] = finish_type == "interrupted"
    t["flags"]["error"] = t["flags"]["error"] or bool(metadata.get("is_error"))

    # The export's own record that a message was never shown, plus the two
    # fields that mark harness text arriving on the user channel. Counting any
    # of them as operator input would inflate typed volume with text nobody
    # wrote, and would open an episode under an instruction nobody gave.
    hidden = bool(metadata.get("is_visually_hidden_from_conversation")
                  or metadata.get("is_user_system_message")
                  or ctype == "user_editable_context")
    t["flags"]["meta"] = role == "user" and hidden

    if role == "user" and not hidden and ctype in TYPED_TYPES and body.strip():
        t["flags"]["typed"] = True
        t["text"] = bounded(body)

    if t["kinds"]["text"]:
        _DIAG.counts["text_messages_emitted"] += 1
    if role == "assistant":
        _DIAG.counts["model_identity_present" if t["model"]
                     else "model_identity_absent"] += 1

    return Turn(**t)


def _digest(message: dict) -> str:
    """sha256 over the message node's canonical serialization.

    A shard is a pretty-printed array, so a message has no source line to hash.
    `records.dumps` sorts every key at every level, which makes this digest
    reproducible from the export without this tool.
    """
    return hashlib.sha256(dumps(message).encode("utf-8")).hexdigest()


def _iso(value: object) -> str:
    """Epoch seconds -> ISO 8601 UTC. Anything else is reported, not repaired."""
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
        except (OSError, OverflowError, ValueError):
            pass
    _DIAG.errors["malformed_timestamp"] += 1
    return ""


def _tools(message: dict, author: dict, source_role: str, role: str, t: dict) -> None:
    """Tool identity, taken from the record's own addressing.

    An assistant message names its target in `recipient`; a tool message names
    itself in `author.name`. Both are structural, so no vocabulary decides what
    counts as a tool call.
    """
    recipient = str(message.get("recipient") or "")
    if role == "assistant" and recipient and recipient != "all":
        t["tools"] = sorted_counts(Counter({recipient: 1}))
        t["kinds"]["tool_use"] = max(t["kinds"]["tool_use"], 1)
    elif source_role == "tool":
        t["tools"] = sorted_counts(Counter({str(author.get("name") or recipient or "?"): 1}))
        t["kinds"]["tool_result"] = max(t["kinds"]["tool_result"], 1)


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------


def _content(ctype: str, content: dict, t: dict) -> str:
    """The ordinary conversation text one content block carries.

    Media is counted and dropped. Nothing is decoded, described, or synthesized
    in its place, so a turn whose only content was an image or a voice note
    reaches the canonical record with zero text characters.
    """
    kinds = t["kinds"]

    if ctype == "text":
        body = _joined(content.get("parts"), t)
        kinds["text"] += 1
        t["chars"]["text"] += len(body)
        return body

    if ctype == "multimodal_text":
        return _multimodal(content.get("parts"), t)

    if ctype == "thoughts":
        kinds["thinking"] += 1
        t["chars"]["thinking"] += _thought_chars(content.get("thoughts"))
        return ""

    if ctype == "reasoning_recap":
        kinds["thinking"] += 1
        t["chars"]["thinking"] += len(str(content.get("content") or ""))
        return ""

    if ctype == "code":
        # Tool input, not prose. Counted as a call so tool volume is comparable
        # with the formats that nest a call inside an assistant message; its
        # length stays out of `chars.text`, which measures conversation text.
        kinds["tool_use"] += 1
        return ""

    if ctype in TOOL_RESULT_TYPES:
        kinds["tool_result"] += 1
        return ""

    if ctype == "system_error":
        kinds["other"] += 1
        t["flags"]["error"] = True
        return ""

    if ctype == "computer_output":
        # A screenshot. Counted as an image and read no further.
        kinds["image"] += 1
        _DIAG.skipped["image"] += 1
        return ""

    if ctype == "user_editable_context":
        # Standing instructions, not a prompt. `_turn` marks it meta.
        kinds["other"] += 1
        return ""

    kinds["other"] += 1
    _DIAG.errors["unknown_content_type"] += 1
    _DIAG.unknown_types[_tag(ctype)] += 1
    return ""


def _joined(parts: object, t: dict) -> str:
    """Text parts, joined. A non-string part here is drift and is counted."""
    kept: list[str] = []
    for part in parts if isinstance(parts, list) else []:
        if isinstance(part, str):
            kept.append(part)
        else:
            _DIAG.errors["non_text_part_in_text_block"] += 1
            t["kinds"]["other"] += 1
    return "\n".join(s for s in kept if s)


def _multimodal(parts: object, t: dict) -> str:
    """Ordinary text from a mixed block. Media components are counted, not read."""
    kept: list[str] = []
    for part in parts if isinstance(parts, list) else []:
        if isinstance(part, str):
            kept.append(part)
            continue
        kind = str(part.get("content_type") or "") if isinstance(part, dict) else ""
        if kind in IMAGE_PARTS:
            t["kinds"]["image"] += 1
            _DIAG.skipped["image"] += 1
        elif kind in AUDIO_PARTS:
            t["kinds"]["other"] += 1
            _DIAG.skipped["audio"] += 1
        else:
            t["kinds"]["other"] += 1
            _DIAG.skipped["other"] += 1
            _DIAG.errors["unknown_part_content_type"] += 1
            _DIAG.unknown_types[_tag(kind)] += 1
    body = "\n".join(s for s in kept if s)
    if kept:
        t["kinds"]["text"] += 1
        t["chars"]["text"] += len(body)
    return body


def _thought_chars(thoughts: object) -> int:
    """Length of the reasoning prose this format records."""
    if not isinstance(thoughts, list):
        return 0
    total = 0
    for item in thoughts:
        if isinstance(item, dict):
            total += len(str(item.get("summary") or ""))
            total += len(str(item.get("content") or ""))
        elif isinstance(item, str):
            total += len(item)
    return total
