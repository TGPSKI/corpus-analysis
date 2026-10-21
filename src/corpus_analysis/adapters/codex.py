"""Codex rollouts: `<root>/YYYY/MM/DD/rollout-<ts>-<session-id>.jsonl`.

Two facts about this format drive the mapping, and a third attributes its
sessions.

**Usage is reported twice, and only one of them is per-turn.** A `token_count`
event carries `total_token_usage`, which is cumulative over the session, and
`last_token_usage`, which is the turn that just finished. Summing the
cumulative one multiplies a session's real spend by its turn count. `last` is
used, with a delta of the cumulative figure as the fallback for releases that
do not carry it.

**A subagent rollout is a session the parent agent opened.** Its first
`session_meta` carries `payload.source` as an object (`{"subagent": …}`)
where operator sessions carry a string (`cli`, `vscode`). Its turns are
flagged `sidechain`, and the messages the parent writes into it are never
marked typed: the operator is not the one who typed them.

**Reasoning text is withheld here too.** A `reasoning` item carries an empty
`summary` and an `encrypted_content` blob, so depth is countable
(`reasoning_output_tokens`, which this format does report) but not readable.

Records map to roles so counts stay comparable with formats that nest tool
calls inside an assistant message: only `agent_message` is an assistant turn.
Tool calls, tool outputs, and reasoning are events that carry their own counts.
"""

from __future__ import annotations

import glob
import re
from collections.abc import Iterator
from pathlib import Path

from ..records import Turn, blank_turn
from .base import SchemaDrift, SessionRef, bounded, parsed

name = "codex"

KNOWN_TYPES = {"session_meta", "turn_context", "event_msg", "response_item"}

#: `apply_patch` takes a patch envelope as a raw string rather than JSON with a
#: path field. Reading it wrong reports zero files edited for a session that
#: rewrote a dozen.
PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", re.M)

REVERT_RE = re.compile(r"\bgit\s+(?:reset|revert|restore|checkout|stash)\b")

EDIT_TOOLS = {"apply_patch", "edit_file", "write_file", "str_replace"}


def default_root() -> Path:
    return Path.home() / ".codex" / "sessions"


def sessions(root: Path | None = None, include_subagents: bool = True) -> list[SessionRef]:
    """Rollouts are filed by date, so a glob finds them without an index."""
    root = Path(root) if root else default_root()
    if not root.is_dir():
        return []
    hits = sorted(glob.glob(str(root / "*" / "*" / "*" / "rollout-*.jsonl")))
    hits += sorted(glob.glob(str(root / "rollout-*.jsonl")))  # flat layouts
    return [SessionRef(name, _session_id(Path(p)), p, project=Path(p).parent.name)
            for p in sorted(set(hits))]


def _session_id(path: Path) -> str:
    """The id is the tail of the filename, and itself contains hyphens.

    Splitting on the last hyphen yields its final segment rather than the id.
    """
    m = re.match(r"rollout-\d{4}-\d{2}-\d{2}T[\d:-]+-(.+)$", path.stem)
    return m.group(1) if m else path.stem


def probe(ref: SessionRef, limit: int = 200) -> None:
    seen: set[str] = set()
    for n, (_i, _sha, rec) in enumerate(parsed(Path(ref.path))):
        seen.add(str(rec.get("type")))
        if seen & KNOWN_TYPES:
            return
        if n >= limit:
            break
    if not seen:
        return
    raise SchemaDrift(
        f"{ref.path}: no known record type in the first {limit} records; "
        f"saw {sorted(seen)[:6]}"
    )


def turns(ref: SessionRef) -> Iterator[Turn]:
    path = Path(ref.path)
    session_id = ref.session_id
    model = version = effort = cwd = ""
    subagent = False
    cumulative = {"input_tokens": 0, "output_tokens": 0,
                  "cache_read_input_tokens": 0, "reasoning_output_tokens": 0}

    for i, sha, rec in parsed(path):
        rtype = str(rec.get("type") or "")
        payload = rec.get("payload") or {}
        ptype = str(payload.get("type") or "")

        if rtype == "session_meta":
            session_id = str(payload.get("session_id") or session_id)
            cwd = str(payload.get("cwd") or cwd)
            version = str(payload.get("cli_version") or version)
            model = str(payload.get("model") or model)
            subagent = _is_subagent_source(payload.get("source"))
        elif rtype == "turn_context":
            model = str(payload.get("model") or model)
            effort = str(payload.get("model_reasoning_effort") or effort)
            cwd = str(payload.get("cwd") or cwd)

        role = "assistant" if ptype == "agent_message" else (
            "user" if ptype == "user_message" else "event")
        t = blank_turn(name, str(path), i, sha, session_id,
                       f"{session_id}:{i}", role, ptype or rtype)
        t["ts"] = str(rec.get("timestamp") or "")
        t["model"], t["version"], t["effort"], t["cwd"] = model, version, effort, cwd
        # Same attribution the Claude Code adapter gives its sidechains: a
        # session the parent agent opened writes its own turns, so every turn
        # is flagged and no user message in it reads as operator-typed.
        t["flags"]["sidechain"] = subagent

        if ptype == "user_message":
            _user(payload, t, subagent)
        elif ptype == "agent_message":
            _agent(payload, t)
        elif ptype == "token_count":
            _usage(payload, t, cumulative)
        elif ptype == "reasoning":
            _reasoning(payload, t)
        elif ptype in ("function_call", "custom_tool_call"):
            _tool_call(payload, t)
        elif ptype in ("function_call_output", "custom_tool_call_output"):
            _tool_output(payload, t)
        elif ptype == "web_search_call":
            t["kinds"]["tool_use"] = 1
            t["tools"] = {"web_search": 1}
        elif ptype == "message":
            # A developer/system prompt replayed into the transcript. Counted
            # as text, never as something the operator typed.
            t["kinds"]["text"] = 1

        yield Turn(**t)


def _is_subagent_source(source: object) -> bool:
    """Whether `session_meta.payload.source` names a subagent session.

    Operator sessions carry a string (`cli`, `vscode`); a session the parent
    agent opened carries an object whose `subagent` key names it. A missing
    `source` (older releases) or an unknown object shape stays an operator
    session: attribution must not drift from one missing field to zero typed
    prompts.
    """
    return isinstance(source, dict) and "subagent" in source


def _user(payload: dict, t: dict, subagent: bool = False) -> None:
    text = str(payload.get("message") or "")
    t["kinds"]["text"] = 1
    t["chars"]["text"] = len(text)
    # In a subagent session the parent agent is the one writing these, so the
    # text still counts as text and never as something the operator typed.
    if text.strip() and not subagent:
        t["flags"]["typed"] = True
        t["text"] = bounded(text)


def _agent(payload: dict, t: dict) -> None:
    text = str(payload.get("message") or "")
    t["kinds"]["text"] = 1
    t["chars"]["text"] = len(text)


def _reasoning(payload: dict, t: dict) -> None:
    t["kinds"]["thinking"] = 1
    t["chars"]["signature"] = len(str(payload.get("encrypted_content") or ""))
    summary = payload.get("summary") or []
    if isinstance(summary, list):
        t["chars"]["thinking"] = sum(
            len(s.get("text", "")) if isinstance(s, dict) else len(str(s))
            for s in summary
        )


def _usage(payload: dict, t: dict, cumulative: dict) -> None:
    """Per-turn usage. `last_token_usage` when present, else a delta."""
    info = payload.get("info") or {}
    last = info.get("last_token_usage")
    total = info.get("total_token_usage") or {}

    if isinstance(last, dict) and last:
        src = last
    else:
        src = {k: max(0, int(total.get(v) or 0) - cumulative[k])
               for k, v in (("input_tokens", "input_tokens"),
                            ("output_tokens", "output_tokens"),
                            ("cache_read_input_tokens", "cached_input_tokens"),
                            ("reasoning_output_tokens", "reasoning_output_tokens"))}

    t["usage"]["input_tokens"] = int(src.get("input_tokens") or 0)
    t["usage"]["output_tokens"] = int(src.get("output_tokens") or 0)
    t["usage"]["cache_read_input_tokens"] = int(
        src.get("cached_input_tokens", src.get("cache_read_input_tokens")) or 0)
    t["usage"]["reasoning_output_tokens"] = int(
        src.get("reasoning_output_tokens") or 0)

    for key, field in (("input_tokens", "input_tokens"),
                       ("output_tokens", "output_tokens"),
                       ("cache_read_input_tokens", "cached_input_tokens"),
                       ("reasoning_output_tokens", "reasoning_output_tokens")):
        cumulative[key] = max(cumulative[key], int(total.get(field) or 0))


def _tool_call(payload: dict, t: dict) -> None:
    tool = str(payload.get("name") or "?")
    t["kinds"]["tool_use"] = 1
    t["tools"] = {tool: 1}
    args = payload.get("arguments") or payload.get("input") or ""
    if not isinstance(args, str) or not args:
        return
    if REVERT_RE.search(args):
        t["flags"]["revert"] = True
    if tool in EDIT_TOOLS or "*** " in args:
        files = {fp.strip(): 1 for fp in PATCH_FILE_RE.findall(args)}
        if files:
            t["files"] = dict(sorted(files.items()))


def _tool_output(payload: dict, t: dict) -> None:
    t["kinds"]["tool_result"] = 1
    out = payload.get("output")
    body = out if isinstance(out, str) else str(out or "")
    # This format reports a failed call in the output body rather than with a
    # flag, so the match is deliberately narrow: a bare "error" substring hits
    # ordinary compiler output and would make every build look broken.
    failed = (
        '"success":false' in body
        or ('"exit_code":' in body and '"exit_code":0' not in body)
    )
    if failed:
        t["flags"]["error"] = True


def first_ts(ref: SessionRef) -> str:
    """The first timestamp, without building a Turn for every record."""
    for _i, _sha, rec in parsed(Path(ref.path)):
        ts = rec.get("timestamp")
        if ts:
            return str(ts)
    return ""
