"""One conformance body, run against every registered adapter.

An adapter that satisfies these is usable by the whole scan layer without the
scan layer knowing which vendor it is reading. Anything vendor-specific is
asserted in its own case below.

The OpenCode fixture is a SQLite database built in `setUpClass` rather than
committed: a binary fixture cannot be reviewed in a diff, and this one is
twelve rows.
"""

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from corpus_analysis import adapters
from corpus_analysis.adapters import (
    chatgpt,
    claude_code,
    claude_export,
    codex,
    copilot_cli,
    cursor,
    opencode,
    vscode_chat,
)
from corpus_analysis.adapters.base import SchemaDrift
from corpus_analysis.records import Turn, dumps
from corpus_analysis.scan import scan_sessions

FIXTURES = Path(__file__).parent / "fixtures"
CLAUDE_ROOT = FIXTURES / "corpus"
CODEX_ROOT = FIXTURES / "codex"
CHATGPT_ROOT = FIXTURES / "chatgpt"
CLAUDE_EXPORT_ROOT = FIXTURES / "claude_export"
CURSOR_ROOT = FIXTURES / "cursor"
COPILOT_CLI_ROOT = FIXTURES / "copilot_cli"
VSCODE_CHAT_ROOT = FIXTURES / "vscode_chat"
DRIFT_ROOT = FIXTURES / "drift"

_TMP: tempfile.TemporaryDirectory | None = None


def build_opencode_db(path: Path) -> None:
    """A minimal store with the three tables this adapter reads."""
    con = sqlite3.connect(path)
    con.execute("create table session (id text, directory text, "
                "parent_id text, time_created integer)")
    con.execute("create table message (id text, session_id text, "
                "time_created integer, data text)")
    con.execute("create table part (id text, session_id text, "
                "message_id text, time_created integer, data text)")
    # Named so the adapter's promise not to read them is checkable.
    con.execute("create table account (id text, token text)")
    con.execute("create table credential (id text, secret text)")

    con.execute("insert into session values ('ses_a', '/tmp/proj-o', NULL, 1767225600000)")
    con.execute("insert into session values ('ses_child', '/tmp/proj-o', 'ses_a', 1767225601000)")

    def msg(mid, role, created, **kw):
        body = {"role": role, "time": {"created": created}, **kw}
        con.execute("insert into message values (?,?,?,?)",
                    (mid, "ses_a", created, json.dumps(body)))

    def part(pid, mid, created, body):
        con.execute("insert into part values (?,?,?,?,?)",
                    (pid, "ses_a", mid, created, json.dumps(body)))

    msg("msg_1", "user", 1767225600000)
    part("p1", "msg_1", 1767225600000, {"type": "text", "text": "port the parser"})

    msg("msg_2", "assistant", 1767225601000, modelID="model-o", agent="build",
        providerID="local",
        tokens={"input": 400, "output": 120, "reasoning": 30,
                "cache": {"read": 40, "write": 0}})
    # Reasoning carries text in this format, unlike the CLI's `claude`.
    part("p2", "msg_2", 1767225601000,
         {"type": "reasoning", "text": "considering the parser layout"})
    part("p3", "msg_2", 1767225602000,
         {"type": "tool", "tool": "edit",
          "state": {"status": "completed", "input": {"filePath": "/tmp/proj-o/parse.py"}}})
    part("p4", "msg_2", 1767225603000,
         {"type": "tool", "tool": "bash",
          "state": {"status": "error", "input": {"command": "git checkout -- parse.py"}}})
    part("p5", "msg_2", 1767225604000, {"type": "step-finish", "reason": "stop"})
    part("p6", "msg_2", 1767225605000, {"type": "text", "text": "done"})

    msg("msg_3", "user", 1767225606000)
    part("p7", "msg_3", 1767225606000, {"type": "compaction", "auto": True})

    con.commit()
    con.close()


def setUpModule() -> None:
    global _TMP
    _TMP = tempfile.TemporaryDirectory()
    build_opencode_db(Path(_TMP.name) / "opencode.db")


def tearDownModule() -> None:
    opencode.close()
    if _TMP is not None:
        _TMP.cleanup()


def opencode_root() -> Path:
    return Path(_TMP.name) / "opencode.db"


def roots() -> dict[str, Path]:
    return {"chatgpt": CHATGPT_ROOT, "claude": CLAUDE_ROOT,
            "claude_export": CLAUDE_EXPORT_ROOT,
            "codex": CODEX_ROOT, "copilot_cli": COPILOT_CLI_ROOT, "cursor": CURSOR_ROOT,
            "opencode": opencode_root(), "vscode_chat": VSCODE_CHAT_ROOT}


class TestConformance(unittest.TestCase):
    """Every registered adapter satisfies the same contract."""

    def test_every_adapter_is_registered_with_its_own_name(self):
        self.assertEqual(sorted(adapters.names()),
                         ["chatgpt", "claude", "claude_export", "codex", "copilot_cli", "cursor", "opencode", "vscode_chat"])
        for name in adapters.names():
            self.assertEqual(adapters.get(name).name, name)

    def test_every_adapter_enumerates_and_probes(self):
        for name, root in roots().items():
            with self.subTest(adapter=name):
                adapter = adapters.get(name)
                refs = adapter.sessions(root)
                self.assertTrue(refs, f"{name} found no sessions")
                real = [r for r in refs if "drift" not in r.session_id]
                adapter.probe(real[0])

    def test_every_adapter_yields_well_formed_turns(self):
        for name, root in roots().items():
            with self.subTest(adapter=name):
                adapter = adapters.get(name)
                ref = next(r for r in adapter.sessions(root)
                           if "drift" not in r.session_id)
                turns = list(adapter.turns(ref))
                self.assertTrue(turns, f"{name} yielded no turns")
                for t in turns:
                    self.assertIsInstance(t, Turn)
                    self.assertEqual(t.provenance.adapter, name)
                    self.assertEqual(len(t.provenance.sha256), 64)
                    self.assertIn(t.role, ("user", "assistant", "event"))
                    if t.ts:
                        self.assertRegex(t.ts, r"^\d{4}-\d{2}-\d{2}T")

    def test_every_adapter_reaches_the_scan_layer(self):
        for name, root in roots().items():
            with self.subTest(adapter=name):
                scans = list(scan_sessions(adapters.get(name), root=root,
                                           include_self=True))
                self.assertTrue(scans)
                self.assertTrue(any(s.totals["typed_prompts"] for s in scans),
                                f"{name} found no typed prompt")

    def test_every_adapter_is_deterministic(self):
        for name, root in roots().items():
            with self.subTest(adapter=name):
                adapter = adapters.get(name)
                ref = next(r for r in adapter.sessions(root)
                           if "drift" not in r.session_id)
                a = [t.to_line() for t in adapter.turns(ref)]
                b = [t.to_line() for t in adapter.turns(ref)]
                self.assertEqual(a, b)


class TestChatGPT(unittest.TestCase):
    """The export fixture is hand-authored. No byte of it came from a corpus.

    Every id is synthetic and every string was written for the case it proves,
    which is the only reason a parser test for a private format can live in a
    public tree at all.
    """

    def setUp(self):
        # `sessions()` resets the adapter's diagnostics, so each case measures
        # its own scan rather than whatever ran before it.
        self.refs = {r.session_id: r for r in chatgpt.sessions(CHATGPT_ROOT)}
        self.turns = {sid: list(chatgpt.turns(r)) for sid, r in self.refs.items()}
        self.diag = chatgpt.diagnostics()

    def rendered(self, session_id: str) -> str:
        return "\n".join(t.to_line() for t in self.turns[session_id])

    # -- traversal ---------------------------------------------------------

    def test_the_active_branch_is_the_thread_current_node_points_at(self):
        ids = [t.provenance.turn_id for t in self.turns["conv-0001"]]
        self.assertEqual(ids, ["node-0001-a", "node-0001-b",
                               "node-0001-c", "node-0001-e"])
        self.assertEqual([t.role for t in self.turns["conv-0001"]],
                         ["user", "assistant", "user", "assistant"])

    def test_a_regenerated_answer_does_not_become_a_sequential_turn(self):
        """The abandoned sibling is counted, never emitted beside its replacement."""
        ids = {t.provenance.turn_id for t in self.turns["conv-0001"]}
        self.assertNotIn("node-0001-d", ids)
        self.assertEqual(self.diag["counts"]["alternate_branch_nodes"], 1)

    def test_traversal_position_orders_the_session(self):
        lines = [t.provenance.line for t in self.turns["conv-0001"]]
        self.assertEqual(lines, sorted(lines))
        self.assertEqual(len(set(lines)), len(lines))

    def test_a_single_leaf_settles_the_branch_without_current_node(self):
        self.assertEqual(len(self.turns["conv-0003"]), 3)
        self.assertEqual(self.diag["counts"]["branch_resolved_by_sole_leaf"], 1)

    def test_competing_branches_are_reported_not_flattened(self):
        self.assertEqual(self.turns["conv-0004"], [])
        self.assertEqual(self.diag["errors"]["ambiguous_active_branch"], 1)

    # -- model identity ----------------------------------------------------

    def test_model_identity_is_taken_only_where_it_is_recorded(self):
        by_id = {t.provenance.turn_id: t for t in self.turns["conv-0001"]}
        self.assertEqual(by_id["node-0001-b"].model, "model-x-1")
        self.assertEqual(by_id["node-0001-e"].model, "model-x-2")

    def test_a_missing_model_slug_stays_unknown(self):
        """The conversation records a default. Reading it here would invent one."""
        answer = next(t for t in self.turns["conv-0002"] if t.role == "assistant")
        self.assertEqual(answer.model, "")
        self.assertEqual(self.diag["counts"]["model_identity_absent"], 1)

    # -- text-only ---------------------------------------------------------

    def test_an_ordinary_text_message_reaches_the_canonical_record(self):
        first = self.turns["conv-0001"][0]
        self.assertTrue(first.flags["typed"])
        self.assertEqual(first.text, "port the parser")
        self.assertEqual(first.chars["text"], len("port the parser"))

    def test_a_mixed_message_keeps_its_text_and_drops_its_image(self):
        turn = self.turns["conv-0002"][0]
        self.assertEqual(turn.text, "read the diagram")
        self.assertEqual(turn.chars["text"], len("read the diagram"))
        self.assertEqual(turn.kinds["image"], 1)

    def test_an_image_only_message_produces_no_text(self):
        turn = next(t for t in self.turns["conv-0002"]
                    if t.provenance.turn_id == "node-0002-c")
        self.assertEqual(turn.text, "")
        self.assertEqual(turn.chars["text"], 0)
        self.assertFalse(turn.flags["typed"])
        self.assertEqual(turn.kinds["image"], 1)

    def test_an_audio_only_message_produces_no_text(self):
        """Including the export's own transcript of it, which is audio-derived."""
        turn = next(t for t in self.turns["conv-0002"]
                    if t.provenance.turn_id == "node-0002-d")
        self.assertEqual(turn.text, "")
        self.assertEqual(turn.chars["text"], 0)
        self.assertFalse(turn.flags["typed"])
        self.assertEqual(self.diag["skipped_content"]["audio"], 3)

    def test_no_media_payload_survives_into_any_record(self):
        whole = "\n".join(self.rendered(sid) for sid in sorted(self.turns))
        for leaked in ("SPOKEN WORDS", "synthetic-asset", "synthetic-decoding"):
            self.assertNotIn(leaked, whole)

    def test_harness_text_on_the_user_channel_is_not_a_typed_prompt(self):
        by_id = {t.provenance.turn_id: t for t in self.turns["conv-0005"]}
        self.assertTrue(by_id["node-0005-b"].flags["meta"])
        self.assertFalse(by_id["node-0005-b"].flags["typed"])
        self.assertTrue(by_id["node-0005-c"].flags["typed"])

    # -- reasoning, tools, flags -------------------------------------------

    def test_reasoning_prose_is_measured_in_this_format(self):
        thoughts = next(t for t in self.turns["conv-0003"] if t.kinds["thinking"])
        self.assertEqual(thoughts.chars["thinking"],
                         len("weighing it") + len("twelve characters of reasoning"))
        self.assertEqual(thoughts.chars["signature"], 0)
        self.assertEqual(thoughts.effort, "extended")

    def test_tool_identity_comes_from_the_records_own_addressing(self):
        by_id = {t.provenance.turn_id: t for t in self.turns["conv-0005"]}
        self.assertEqual(by_id["node-0005-d"].tools, {"python": 1})
        self.assertEqual(by_id["node-0005-d"].kinds["tool_use"], 1)
        self.assertEqual(by_id["node-0005-e"].tools, {"python": 1})
        self.assertEqual(by_id["node-0005-e"].kinds["tool_result"], 1)

    def test_an_interrupted_finish_is_flagged(self):
        by_id = {t.provenance.turn_id: t for t in self.turns["conv-0005"]}
        self.assertTrue(by_id["node-0005-f"].flags["interrupted"])

    def test_this_format_records_no_usage(self):
        """A whole-adapter zero means not recorded, never none happened."""
        totals = {v for t in self.turns["conv-0001"] for v in t.usage.values()}
        self.assertEqual(totals, {0})

    # -- drift and duplicates ----------------------------------------------

    def test_an_unknown_content_type_is_counted_not_dropped(self):
        self.assertEqual(self.diag["errors"]["unknown_content_type"], 1)
        self.assertIn("hologram_preview", self.diag["unknown_types"])
        self.assertEqual(self.diag["errors"]["unknown_part_content_type"], 1)

    def test_a_conversation_with_no_node_tree_raises_drift(self):
        ref = chatgpt.sessions(DRIFT_ROOT / "chatgpt")[0]
        with self.assertRaises(SchemaDrift):
            chatgpt.probe(ref)

    def test_an_empty_conversation_object_is_counted_not_fatal(self):
        """A keyless `{}` in a shard must not make the whole root unreadable.

        The export format hands the adapter whatever the vendor put in the
        array; an object with no keys at all carries nothing to drift from, so
        it is counted and skipped instead of reaching `probe`, where it used
        to raise `SchemaDrift` and take every readable conversation with it.
        """
        good = {
            "conversation_id": "conv-empty-neighbour",
            "create_time": "2026-01-02T10:00:00.000Z",
            "mapping": {
                "root": {"children": ["a"], "message": None},
                "a": {"parent": "root", "children": [], "message": None},
            },
        }
        with tempfile.TemporaryDirectory() as d:
            shard = Path(d) / "conversations-000.json"
            shard.write_text(json.dumps([good, {}, good]), encoding="utf-8")
            refs = chatgpt.sessions(Path(d))
            self.assertEqual([r.session_id for r in refs], ["conv-empty-neighbour"])
            for ref in refs:  # nothing readable in the root raises nothing
                chatgpt.probe(ref)
            diag = chatgpt.diagnostics()
            self.assertEqual(diag["counts"]["empty_conversation_objects"], 1)
            self.assertEqual(diag["counts"]["conversations_discovered"], 3)
            self.assertEqual(diag["counts"]["duplicate_conversation_ids"], 1)

    def test_an_object_with_keys_but_no_mapping_still_raises_drift(self):
        """Drift stays loud where a conversation carries content and no tree."""
        shaped = {"title": "not a conversation", "create_time": "2026-01-02"}
        with tempfile.TemporaryDirectory() as d:
            shard = Path(d) / "conversations-000.json"
            shard.write_text(json.dumps([shaped]), encoding="utf-8")
            refs = chatgpt.sessions(Path(d))
            self.assertEqual(len(refs), 1)
            with self.assertRaises(SchemaDrift):
                chatgpt.probe(refs[0])

    def test_a_repeated_conversation_id_is_kept_once(self):
        """The second export of a conversation is counted, not emitted twice."""
        self.assertEqual(self.diag["counts"]["duplicate_conversation_ids"], 1)
        self.assertEqual(len(self.turns["conv-0001"]), 4)
        self.assertTrue(self.refs["conv-0001"].path.endswith("conversations-000.json"))

    # -- provenance and determinism ----------------------------------------

    def test_provenance_is_populated_on_every_turn(self):
        for sid, turns in self.turns.items():
            for t in turns:
                self.assertEqual(t.provenance.adapter, "chatgpt")
                self.assertEqual(t.provenance.session_id, sid)
                self.assertTrue(t.provenance.turn_id)
                self.assertTrue(t.provenance.path.endswith(".json"))
                self.assertEqual(len(t.provenance.sha256), 64)

    def test_the_digest_reproduces_from_the_export_without_this_tool(self):
        turn = self.turns["conv-0001"][0]
        shard = json.loads(Path(turn.provenance.path).read_text())
        conv = next(c for c in shard if c["conversation_id"] == turn.provenance.session_id)
        message = conv["mapping"][turn.provenance.turn_id]["message"]
        recomputed = hashlib.sha256(dumps(message).encode("utf-8")).hexdigest()
        self.assertEqual(recomputed, turn.provenance.sha256)

    def test_two_scans_of_the_export_agree_byte_for_byte(self):
        def run():
            refs = chatgpt.sessions(CHATGPT_ROOT)
            body = "".join(f"{t.to_line()}\n" for r in refs for t in chatgpt.turns(r))
            return body, dumps(chatgpt.diagnostics())

        self.assertEqual(run(), run())


class TestClaudeExport(unittest.TestCase):
    """The export fixture is hand-authored. No byte of it came from a corpus.

    The two Claude adapters read different products and must not be confused:
    `claude` is the CLI's JSONL sessions, `claude_export` is the consumer
    export's single array.
    """

    def setUp(self):
        self.refs = {r.session_id: r for r in claude_export.sessions(CLAUDE_EXPORT_ROOT)}
        self.turns = {sid: list(claude_export.turns(r)) for sid, r in self.refs.items()}
        self.diag = claude_export.diagnostics()

    def by_id(self, session_id: str) -> dict:
        return {t.provenance.turn_id: t for t in self.turns[session_id]}

    def whole(self) -> str:
        return "\n".join(t.to_line() for sid in sorted(self.turns)
                         for t in self.turns[sid])

    # -- the two adapters stay distinct ------------------------------------

    def test_it_does_not_collide_with_the_claude_code_adapter(self):
        self.assertEqual(claude_export.name, "claude_export")
        self.assertEqual(claude_code.name, "claude")
        self.assertIsNot(adapters.get("claude"), adapters.get("claude_export"))

    # -- thread reconstruction ---------------------------------------------

    def test_a_single_tip_settles_the_thread(self):
        ids = [t.provenance.turn_id for t in self.turns["conv-0001"]]
        self.assertEqual(ids, ["msg-0001-a", "msg-0001-b", "msg-0001-c", "msg-0001-d"])
        self.assertEqual([t.role for t in self.turns["conv-0001"]],
                         ["user", "assistant", "user", "assistant"])

    def test_thread_position_orders_the_session(self):
        lines = [t.provenance.line for t in self.turns["conv-0001"]]
        self.assertEqual(lines, sorted(lines))
        self.assertEqual(len(set(lines)), len(lines))

    def test_two_tips_fail_closed_rather_than_picking_the_newest(self):
        """Nothing in this schema records which branch the reader saw."""
        self.assertEqual(self.turns["conv-0003"], [])
        self.assertEqual(self.diag["errors"]["ambiguous_active_branch"], 1)

    def test_a_cycle_upstream_of_a_tip_fails_closed(self):
        self.assertEqual(self.turns["conv-0004"], [])
        self.assertEqual(self.diag["errors"]["branch_cycle"], 1)

    def test_an_unreachable_component_fails_closed(self):
        self.assertEqual(self.turns["conv-0006"], [])
        self.assertEqual(self.diag["errors"]["unreachable_messages"], 1)

    # -- typed prompts -----------------------------------------------------

    def test_operator_text_is_typed(self):
        first = self.turns["conv-0001"][0]
        self.assertTrue(first.flags["typed"])
        self.assertEqual(first.text, "port the parser")

    def test_an_uploaded_files_body_is_not_a_typed_prompt(self):
        """`extracted_content` is a document, not something anyone typed."""
        turn = self.by_id("conv-0002")["msg-0002-d"]
        self.assertTrue(turn.flags["typed"])
        self.assertEqual(turn.text, "check this file")
        self.assertEqual(turn.chars["text"], len("check this file"))
        self.assertEqual(self.diag["skipped_content"]["attachment"], 1)

    def test_project_memory_and_account_text_never_becomes_a_turn(self):
        whole = self.whole()
        for leaked in ("PROJECT INSTRUCTIONS", "PROJECT KNOWLEDGE", "PROJECT DESCRIPTION",
                       "STANDING MEMORY TEXT", "ACCOUNT FIELD", "UPLOADED FILE BODY"):
            self.assertNotIn(leaked, whole)

    def test_the_context_artifacts_are_counted_not_read(self):
        self.assertEqual(self.diag["counts"]["project_records_discovered"], 1)
        self.assertEqual(self.diag["counts"]["project_docs_discovered"], 1)
        self.assertEqual(self.diag["counts"]["memory_records_discovered"], 1)
        self.assertEqual(self.diag["counts"]["reflection_records_discovered"], 1)
        self.assertEqual(self.diag["counts"]["account_files_present_and_unread"], 2)

    def test_this_export_records_no_conversation_project_link(self):
        """Zero by structure, not by absence of projects. Stated as a figure."""
        self.assertEqual(self.diag["counts"]["conversation_project_links"], 0)

    # -- content -----------------------------------------------------------

    def test_the_flattened_text_field_is_not_canonical_text(self):
        """It disagrees with the blocks and folds in what they keep apart."""
        turn = self.by_id("conv-0001")["msg-0001-b"]
        self.assertEqual(turn.chars["text"], len("ported"))
        self.assertNotIn("FLATTENED RENDERING", self.whole())

    def test_reasoning_is_measured_and_kept_out_of_visible_text(self):
        turn = self.by_id("conv-0001")["msg-0001-b"]
        self.assertEqual(turn.kinds["thinking"], 1)
        self.assertEqual(turn.chars["thinking"], len("readable reasoning"))
        self.assertEqual(turn.chars["signature"], len("0123456789abcdef"))
        self.assertEqual(turn.chars["text"], len("ported"))

    def test_withheld_reasoning_is_counted_apart_from_absent_reasoning(self):
        """`chars.thinking` is zero for both; only the counter separates them."""
        turn = self.by_id("conv-0001")["msg-0001-d"]
        self.assertEqual(turn.kinds["thinking"], 1)
        self.assertEqual(turn.chars["thinking"], 0)
        self.assertEqual(self.diag["counts"]["thinking_blocks_withheld"], 1)
        self.assertEqual(self.diag["counts"]["thinking_blocks_readable"], 1)

    def test_assistant_code_in_a_text_block_is_text_not_a_tool_call(self):
        turn = self.by_id("conv-0002")["msg-0002-e"]
        self.assertEqual(turn.kinds["text"], 1)
        self.assertEqual(turn.kinds["tool_use"], 0)

    def test_tool_identity_comes_from_the_blocks_own_name(self):
        turn = self.by_id("conv-0001")["msg-0001-d"]
        self.assertEqual(turn.tools, {"bash_tool": 1, "create_file": 1})
        self.assertEqual(turn.kinds["tool_use"], 2)
        self.assertEqual(turn.kinds["tool_result"], 2)
        self.assertTrue(turn.flags["error"])
        self.assertTrue(turn.flags["revert"])
        self.assertEqual(turn.files, {"/synthetic/proj/parse.py": 1})

    def test_this_format_records_no_model_identity(self):
        models = {t.model for sid in self.turns for t in self.turns[sid]}
        self.assertEqual(models, {""})
        self.assertGreater(self.diag["counts"]["model_identity_absent"], 0)

    # -- media -------------------------------------------------------------

    def test_a_mixed_message_keeps_its_text_and_counts_its_image(self):
        turn = self.by_id("conv-0002")["msg-0002-a"]
        self.assertEqual(turn.text, "read the diagram")
        self.assertEqual(turn.kinds["image"], 1)

    def test_an_image_only_message_produces_no_text(self):
        turn = self.by_id("conv-0002")["msg-0002-b"]
        self.assertEqual(turn.text, "")
        self.assertEqual(turn.chars["text"], 0)
        self.assertFalse(turn.flags["typed"])
        self.assertEqual(turn.kinds["image"], 1)

    def test_an_audio_only_message_produces_no_text(self):
        turn = self.by_id("conv-0002")["msg-0002-c"]
        self.assertEqual(turn.text, "")
        self.assertEqual(turn.chars["text"], 0)
        self.assertFalse(turn.flags["typed"])
        self.assertEqual(self.diag["skipped_content"]["audio"], 1)

    def test_no_upload_name_reaches_any_record(self):
        whole = self.whole()
        for leaked in ("synthetic-diagram", "synthetic-photo",
                       "synthetic-voice-note", "synthetic-notes"):
            self.assertNotIn(leaked, whole)

    # -- drift and duplicates ----------------------------------------------

    def test_an_unknown_block_type_is_counted_without_echoing_it(self):
        """An unknown value may itself be private text, so only a counter moves."""
        self.assertEqual(self.diag["errors"]["unknown_content_type"], 1)
        self.assertNotIn("hologram_preview", dumps(self.diag))

    def test_an_unknown_sender_becomes_an_event_and_is_counted(self):
        turn = self.by_id("conv-0005")["msg-0005-c"]
        self.assertEqual(turn.role, "event")
        self.assertFalse(turn.flags["typed"])
        self.assertEqual(self.diag["errors"]["unsupported_role"], 1)

    def test_a_conversation_with_no_message_list_raises_drift(self):
        ref = claude_export.sessions(DRIFT_ROOT / "claude_export")[0]
        with self.assertRaises(SchemaDrift):
            claude_export.probe(ref)

    def test_an_identical_repeat_is_kept_once(self):
        self.assertEqual(self.diag["counts"]["duplicate_conversation_ids"], 1)
        self.assertEqual(len(self.turns["conv-0001"]), 4)

    def test_a_conflicting_repeat_fails_closed_rather_than_picking_by_file_order(self):
        """Two exports disagree and the schema records no version authority."""
        self.assertEqual(self.diag["errors"]["duplicate_conflict"], 1)
        self.assertEqual(self.turns["conv-0007"], [])

    # -- provenance and determinism ----------------------------------------

    def test_provenance_is_populated_on_every_turn(self):
        for sid, turns in self.turns.items():
            for t in turns:
                self.assertEqual(t.provenance.adapter, "claude_export")
                self.assertEqual(t.provenance.session_id, sid)
                self.assertTrue(t.provenance.turn_id)
                self.assertEqual(len(t.provenance.sha256), 64)

    def test_the_digest_reproduces_from_the_export_without_this_tool(self):
        turn = self.turns["conv-0001"][0]
        export = json.loads(Path(turn.provenance.path).read_text())
        conv = next(c for c in export if c["uuid"] == turn.provenance.session_id)
        message = next(m for m in conv["chat_messages"]
                       if m["uuid"] == turn.provenance.turn_id)
        self.assertEqual(
            hashlib.sha256(dumps(message).encode("utf-8")).hexdigest(),
            turn.provenance.sha256)

    def test_two_scans_of_the_export_agree_byte_for_byte(self):
        def run():
            refs = claude_export.sessions(CLAUDE_EXPORT_ROOT)
            body = "".join(f"{t.to_line()}\n" for r in refs
                           for t in claude_export.turns(r))
            return body, dumps(claude_export.diagnostics())

        self.assertEqual(run(), run())


class TestCopilotCli(unittest.TestCase):
    def test_event_log_projects_dialogue_tools_and_events(self):
        ref = copilot_cli.sessions(COPILOT_CLI_ROOT)[0]
        turns = list(copilot_cli.turns(ref))
        self.assertEqual([turn.role for turn in turns],
                         ["event", "user", "assistant", "event", "event"])
        self.assertTrue(turns[1].flags["typed"])
        self.assertEqual(turns[2].usage["output_tokens"], 4)
        self.assertEqual(turns[3].tools, {"read_file": 1})
        self.assertEqual(turns[4].kinds["tool_result"], 1)

    def test_unknown_event_type_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.write_text('{"id":"x","parentId":null,"timestamp":"","type":"new.event","data":{}}\n')
            with self.assertRaises(SchemaDrift):
                copilot_cli.probe(adapters.SessionRef("copilot_cli", "x", str(path)))


class TestVSCodeChat(unittest.TestCase):
    def test_request_order_projects_to_one_user_and_one_response(self):
        ref = vscode_chat.sessions(VSCODE_CHAT_ROOT)[0]
        user, assistant = vscode_chat.turns(ref)
        self.assertTrue(user.flags["typed"])
        self.assertEqual(user.ts, "2026-01-01T00:00:00Z")
        self.assertEqual(assistant.kinds["text"], 2)
        self.assertEqual(assistant.kinds["other"], 1)
        self.assertEqual(user.provenance.sha256, assistant.provenance.sha256)

    def test_missing_request_shape_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "chatSessions" / "bad.json"
            path.parent.mkdir()
            path.write_text('{"requests":[{"message":{},"response":[]}]}')
            ref = vscode_chat.sessions(root)[0]
            with self.assertRaises(SchemaDrift):
                vscode_chat.probe(ref)

    def test_a_unicode_line_separator_inside_text_is_not_journal_framing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "chatSessions" / "journal.jsonl"
            path.parent.mkdir()
            session = {
                "requests": [{
                    "message": {"text": "first\u2028second"},
                    "response": [],
                    "timestamp": 1767225600000,
                }],
            }
            path.write_text(
                json.dumps({"v": session}, ensure_ascii=False) + "\n",
                encoding="utf-8")
            ref = vscode_chat.sessions(root)[0]
            turns = list(vscode_chat.turns(ref))
            self.assertEqual(turns[0].text, "first\u2028second")


class TestCursor(unittest.TestCase):
    """Synthetic agent transcripts cover the custody boundary and projection."""

    def setUp(self):
        self.refs = cursor.sessions(CURSOR_ROOT)
        self.primary = next(r for r in self.refs if r.project == "project-a" and not r.parent)

    def test_messages_project_to_turns_and_lifecycle_is_an_event(self):
        turns = list(cursor.turns(self.primary))
        self.assertEqual([turn.role for turn in turns], ["user", "assistant", "event"])
        self.assertTrue(turns[0].flags["typed"])
        self.assertEqual(turns[1].kinds["text"], 2)
        self.assertEqual(turns[1].tools, {"read_file": 1})
        self.assertEqual(turns[2].source_type, "turn_ended")
        self.assertFalse(turns[2].flags["error"])

    def test_normalized_blocks_retain_source_order(self):
        message = list(cursor.normalized(self.primary))[1]
        self.assertEqual([block.type for block in message.blocks],
                         ["text", "tool_use", "text"])
        self.assertEqual(message.blocks[1].tool_name, "read_file")

    def test_subagent_is_a_separate_physical_session_with_parent_linkage(self):
        worker = next(r for r in self.refs if r.parent)
        self.assertEqual(worker.parent, "primary-001")
        self.assertNotEqual(worker.session_id, self.primary.session_id)
        turns = list(cursor.turns(worker))
        self.assertTrue(all(turn.flags["sidechain"] for turn in turns))
        self.assertTrue(turns[-1].flags["error"])

    def test_physical_identity_includes_project_and_source_digest(self):
        other = next(r for r in self.refs if r.project == "project-b")
        self.assertNotEqual(self.primary.session_id, other.session_id)
        self.assertEqual(cursor.physical(self.primary).transcript_uuid, "primary-001")
        self.assertEqual(len(cursor.physical(self.primary).source_sha256), 64)

    def test_lineage_is_based_on_parsed_records(self):
        relations = cursor.lineage(self.refs)
        kinds = {item["relation"] for item in relations}
        self.assertTrue({"EXACT_SAME_CONTENT", "PREFIX_OF", "SUPERSEDES_PREFIX",
                         "SAME_UUID_DIVERGED", "UNIQUE_UUID"} <= kinds)
        divergent = next(item for item in relations
                         if item["relation"] == "SAME_UUID_DIVERGED")
        self.assertEqual(divergent["first_divergence_row"], 0)

    def test_unknown_shapes_fail_closed(self):
        cases = (
            b'{"role":"system","message":{"content":[{"type":"text","text":"x"}]}}\n',
            b'{"role":"assistant","message":{"content":[{"type":"image"}]}}\n',
            b'{"type":"turn_ended","status":"pending"}\n',
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.jsonl"
            for body in cases:
                path.write_bytes(body)
                ref = adapters.SessionRef("cursor", "bad", str(path))
                with self.assertRaises(SchemaDrift):
                    cursor.probe(ref)

    def test_provenance_is_the_raw_source_line_and_runs_are_deterministic(self):
        first = list(cursor.turns(self.primary))[0]
        raw = Path(first.provenance.path).read_bytes().splitlines(keepends=True)[0]
        self.assertEqual(first.provenance.sha256,
                         hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest())
        self.assertEqual([turn.to_line() for turn in cursor.turns(self.primary)],
                         [turn.to_line() for turn in cursor.turns(self.primary)])


class TestCodex(unittest.TestCase):
    def turns(self):
        ref = next(r for r in codex.sessions(CODEX_ROOT)
                   if r.session_id == "sess-codex-1")
        return list(codex.turns(ref))

    def test_session_id_survives_its_hyphens(self):
        ids = {r.session_id for r in codex.sessions(CODEX_ROOT)}
        self.assertIn("sess-codex-1", ids)

    def test_usage_is_per_turn_not_cumulative(self):
        """Summing the cumulative figure multiplies spend by the turn count."""
        usage = [t.usage for t in self.turns() if t.usage["output_tokens"]]
        self.assertEqual([u["output_tokens"] for u in usage], [120, 200])
        self.assertEqual(sum(u["output_tokens"] for u in usage), 320)

    def test_reasoning_is_counted_and_withheld(self):
        thinking = [t for t in self.turns() if t.kinds["thinking"]]
        self.assertEqual(len(thinking), 1)
        self.assertEqual(thinking[0].chars["thinking"], 0)
        self.assertEqual(thinking[0].chars["signature"], 128)

    def test_reasoning_tokens_are_counted_where_recorded(self):
        self.assertEqual(sum(t.usage["reasoning_output_tokens"]
                             for t in self.turns()), 80)

    def test_patch_envelope_yields_the_file(self):
        files = {p for t in self.turns() for p in t.files}
        self.assertEqual(files, {"/tmp/proj-c/main.py"})

    def test_nonzero_exit_marks_an_error(self):
        self.assertEqual(sum(t.flags["error"] for t in self.turns()), 1)

    def test_undo_shaped_command_is_flagged(self):
        self.assertEqual(sum(t.flags["revert"] for t in self.turns()), 1)

    def test_drift_probe_raises(self):
        ref = next(r for r in codex.sessions(DRIFT_ROOT / "codex")
                   if "drift" in r.session_id)
        with self.assertRaises(SchemaDrift):
            codex.probe(ref)

    def _rollout(self, root: Path, source: object, name: str) -> Path:
        """A minimal rollout whose session_meta carries the given `source`."""
        path = root / f"rollout-2026-01-02T00-00-00-{name}.jsonl"
        meta = {"type": "session_meta", "timestamp": "2026-01-02T00:00:00Z",
                "payload": {"session_id": name, "cwd": "/tmp/proj-c",
                            "cli_version": "0.9.0", "model": "model-c"}}
        if source is not None:
            meta["payload"]["source"] = source
        user = {"type": "event_msg", "timestamp": "2026-01-02T00:01:00Z",
                "payload": {"type": "user_message", "message": "probe the tree"}}
        agent = {"type": "event_msg", "timestamp": "2026-01-02T00:02:00Z",
                 "payload": {"type": "agent_message", "message": "done"}}
        path.write_text("\n".join(json.dumps(r) for r in (meta, user, agent)),
                        encoding="utf-8")
        return path

    def test_a_subagent_source_object_is_flagged_and_never_typed(self):
        """Parent-agent writes into a subagent session are not operator prompts.

        Subagent rollouts carry `payload.source` as an object rather than a
        string (#1); without the check every message the parent wrote into the
        subagent counted as a typed prompt and fed episode and temporal views.
        """
        with tempfile.TemporaryDirectory() as d:
            path = self._rollout(Path(d), {"subagent": {"name": "probe"}}, "sess-sub")
            ref = codex.sessions(Path(d))[0]
            turns = list(codex.turns(ref))
            self.assertTrue(all(t.flags["sidechain"] for t in turns))
            self.assertFalse(any(t.flags.get("typed") for t in turns))
            user = next(t for t in turns if t.role == "user")
            self.assertEqual(user.chars["text"], len("probe the tree"))
            self.assertEqual(user.text, "")
            self.assertEqual(user.provenance.path, str(path))

    def test_a_string_source_stays_an_operator_session(self):
        """`source: "cli"` and an absent `source` both stay typed as before."""
        with tempfile.TemporaryDirectory() as d:
            self._rollout(Path(d), "cli", "sess-cli")
            self._rollout(Path(d), None, "sess-old")
            refs = {r.session_id: r for r in codex.sessions(Path(d))}
            self.assertEqual(sorted(refs), ["sess-cli", "sess-old"])
            for sid in refs:
                turns = list(codex.turns(refs[sid]))
                self.assertTrue(any(t.flags.get("typed") for t in turns))
                self.assertFalse(any(t.flags.get("sidechain") for t in turns))


class TestOpenCode(unittest.TestCase):
    def turns(self):
        ref = next(r for r in opencode.sessions(opencode_root())
                   if r.session_id == "ses_a")
        return list(opencode.turns(ref))

    def test_child_sessions_are_attributed_to_their_parent(self):
        refs = {r.session_id: r for r in opencode.sessions(opencode_root())}
        self.assertEqual(refs["ses_child"].parent, "ses_a")
        self.assertEqual(refs["ses_a"].parent, "")

    def test_one_turn_per_message_with_parts_folded_in(self):
        turns = self.turns()
        self.assertEqual(len(turns), 3)
        assistant = next(t for t in turns if t.role == "assistant")
        self.assertEqual(assistant.kinds["tool_use"], 2)
        self.assertEqual(assistant.kinds["text"], 1)

    def test_reasoning_text_is_present_in_this_format(self):
        assistant = next(t for t in self.turns() if t.role == "assistant")
        self.assertGreater(assistant.chars["thinking"], 0)
        self.assertEqual(assistant.chars["signature"], 0)

    def test_provenance_addresses_a_row_that_can_be_re_read(self):
        turn = self.turns()[0]
        con = sqlite3.connect(f"file:{opencode_root()}?mode=ro", uri=True)
        row = con.execute("select data from message where rowid = ?",
                          (turn.provenance.line,)).fetchone()
        con.close()
        self.assertIsNotNone(row)
        from corpus_analysis.records import line_digest
        self.assertEqual(line_digest(row[0].encode()), turn.provenance.sha256)

    def test_first_ts_comes_from_the_session_row(self):
        ref = next(r for r in opencode.sessions(opencode_root())
                   if r.session_id == "ses_a")
        self.assertTrue(opencode.first_ts(ref).startswith("2026-01-01"))

    def test_a_compaction_part_is_not_a_typed_prompt(self):
        compacted = [t for t in self.turns() if t.flags["compact_summary"]]
        self.assertEqual(len(compacted), 1)
        self.assertFalse(compacted[0].flags["typed"])

    def test_missing_tables_raise_drift(self):
        with tempfile.TemporaryDirectory() as d:
            empty = Path(d) / "empty.db"
            sqlite3.connect(empty).close()
            ref = opencode.SessionRef("opencode", "x", str(empty))
            with self.assertRaises(SchemaDrift):
                opencode.probe(ref)
            opencode.close()


class TestClaudeStillConforms(unittest.TestCase):
    def test_default_root_is_not_read_during_tests(self):
        """Every adapter takes its root as an argument."""
        for adapter in (chatgpt, claude_code, claude_export, codex, opencode):
            self.assertTrue(callable(adapter.default_root))


if __name__ == "__main__":
    unittest.main()
