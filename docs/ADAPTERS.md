# Adapter reference

Each adapter converts one vendor format to the canonical `Turn` and
`Provenance` records. The scan layer does not inspect vendor fields. A probe
raises `SchemaDrift` when the source no longer has the structure the adapter
knows, so a format change cannot silently look like an empty corpus.

The conformance tests in `tests/test_adapters.py` exercise every registered
adapter. The fixture paths below identify the synthetic coverage; they are not
copies of vendor transcripts.

## Coverage matrix

| Name | Source shape | Default root | Fixture coverage | Source-specific loss or caution |
|---|---|---|---|---|
| `claude` | Claude Code JSONL sessions and `subagents/agent-*.jsonl` | `~/.claude/projects` | `tests/fixtures/corpus/` | Parent and subagent sessions are separate records with parent attribution. Continuation summaries are harness-authored and do not count as typed prompts. |
| `codex` | dated or flat `rollout-*.jsonl` | `~/.codex/sessions` | `tests/fixtures/codex/` | Cumulative usage is converted to per-turn deltas. A cumulative field must not be summed again downstream. |
| `opencode` | SQLite `session`, `message`, and `part` tables | `~/.local/share/opencode/opencode.db` | built by `tests/test_adapters.py` at test time; no binary fixture is committed | Reads the database through a read-only connection. Child sessions are retained and linked to their parent. |
| `chatgpt` | `conversations*.json` export shards containing conversation trees | `~/corpus/chatgpt` | `tests/fixtures/chatgpt/export-synthetic/` | Traverses the branch named by `current_node`; ambiguous branches emit diagnostics and no turns. Image and audio content is counted and dropped. |
| `claude_export` | `conversations.json` export arrays | `~/corpus/claude` | `tests/fixtures/claude_export/export-synthetic/` | No active-branch pointer exists. Multiple tips, cycles, and unreachable components fail closed. Memories, projects, reflections, and account files are structural inputs, not turns. |
| `copilot_cli` | recursive per-session `events.jsonl` | `~/.copilot` | `tests/fixtures/copilot_cli/` | The session-local event log is authoritative. Global stores and todo databases are not merged. Unknown event types raise drift. |
| `cursor` | JSONL agent transcripts under `agent-transcripts/` | `~/.cursor/projects` | `tests/fixtures/cursor/` | Primary and subagent transcripts are separate physical sessions. Identity includes project, transcript id, and source digest because the same transcript id can be rewritten. |
| `vscode_chat` | `chatSessions/*.json` and storage-journal JSONL | `~/.config/Code/User/workspaceStorage` | `tests/fixtures/vscode_chat/` | Journal state is selected only when exactly one session-state value is present. Physical source identity changes when the session file changes. |

## Shared canonical fields

Every emitted `Turn` has these top-level fields:

```text
schema_version, role, source_type, ts, model, version, effort, cwd,
kinds, chars, usage, invariants, tools, files, flags, text, provenance
```

Adapters fill only values their source records. A zero or empty value can mean
“not recorded”; it does not mean that the source established absence. In
particular, reasoning may be represented as a visible thinking block, a
withheld signature, a token count, or nothing at all depending on the adapter.

`text` is typed prompt text only. Assistant bodies are bounded to measurements
such as character counts. `provenance` identifies the adapter, source path,
zero-based line or reconstructed position, digest, session id, and turn id.

## Choosing an adapter

Use the adapter whose source semantics match the bytes, not the vendor name
alone:

- Use `claude` for Claude Code session logs and `claude_export` for the Claude
  consumer data export. They are different formats and are not interchangeable.
- Use `chatgpt` only for ChatGPT export shards. Its branch pointer makes a
  visible thread recoverable; do not flatten `mapping` or sort it by timestamp.
- Use `opencode` for its SQLite store, not a copied database table dump. The
  adapter needs the session and part relationships to reconstruct turns.
- Use `copilot_cli`, `cursor`, or `vscode_chat` only when the source has the
  corresponding local event/transcript structure. A file being JSONL does not
  make it Claude Code or Codex JSONL.

When a source is a private export, pass an explicit `--root`, use an absolute
time window, and keep output under an ignored local directory. See
[SECURITY.md](../SECURITY.md) and the [user guide](USER_GUIDE.md).

## Drift and diagnostics

Drift is an input failure, not a low-count result. Preserve the failing source
and the adapter name, then inspect the vendor format before changing the
adapter. `stats` may also report structural diagnostics such as ambiguous
branches, duplicate ids, skipped media, or malformed records. An object with
no keys at all (an empty `{}` in a conversation array) is not drift: the
ChatGPT adapter counts it as `empty_conversation_objects` entries and reads the
rest of the shard. Diagnostics are
aggregate categories; they must not echo source values.

The adapter boundary is also the downstream contract. A consumer such as
`flow-indicator` should receive canonical ingest JSONL or an explicitly
documented conversion, never a vendor file whose format it does not own. The
three formats that `flow-indicator` reads natively (`claude-code`, `codex`, and
`opencode`) can be sent to it directly; the other five require its `generic`
bridge or a native adapter there. See [USER_GUIDE.md](USER_GUIDE.md#handoff-to-flow-indicator-or-another-downstream-tool).
