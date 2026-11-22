"""One Astra turn = one `codex exec` call, with every built-in Codex tool switched off.

This mirrors the harness of the MIT weight-space-programming logs (external/lm_understand-logs/harness):
the model gets no tools of its own, the prompt goes in on stdin, and each turn's raw event stream is
audited so a turn only counts when it holds nothing but the model's messages and reasoning.
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
INSTRUCTIONS = HARNESS_DIR / "instructions_neutral.md"

# The standalone CLI (0.144.x) is too old for gpt-6-astra; the build bundled with ChatGPT.app is not.
DEFAULT_CODEX = "/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex"
CODEX = os.environ.get("CODEX_BIN", DEFAULT_CODEX)
MODEL = "gpt-6-astra"

# Every feature that hands the model a tool (the MIT list plus the ones newer builds added).
DISABLED_FEATURES = [
    "apps", "browser_use", "browser_use_external", "browser_use_full_cdp_access", "code_mode",
    "code_mode_host", "code_mode_host_grpc", "code_mode_interrupt", "collaboration_modes", "computer_use",
    "default_mode_request_user_input", "goals", "hooks", "image_generation", "in_app_browser",
    "in_app_chat", "in_app_local_automation", "multi_agent", "multi_agent_v2", "plugins",
    "plugin_sharing", "realtime_conversation", "remote_plugin", "shell_snapshot", "shell_tool",
    "skill_mcp_dependency_install", "skill_search", "sleep_tool", "tool_suggest", "unified_exec",
    "unified_exec_tty", "view_image", "workspace_dependencies", "worktrees",
]  # fmt: skip

# Event-stream item types that are not tool use.
NON_TOOL_ITEMS = {"agent_message", "reasoning", "error"}

REPO = HARNESS_DIR.parents[1]


def redact(text: str) -> str:
    """Replace machine-specific paths with placeholders before anything is written to a log."""
    return text.replace(str(REPO), "<repo>").replace(str(Path.home()), "<home>")


@dataclass
class TurnResult:
    ok: bool
    answer: str
    thread_id: str | None
    tool_free: bool
    usage: dict = field(default_factory=dict)
    wall_s: float = 0.0
    rc: int = 0
    errors: list = field(default_factory=list)
    item_types: dict = field(default_factory=dict)
    cmd: list = field(default_factory=list)


def _config_args(effort: str) -> list[str]:
    args = [
        "-c", f'model_reasoning_effort="{effort}"',
        "-c", 'approval_policy="never"',
        "-c", 'web_search="disabled"',
        "-c", f'model_instructions_file="{INSTRUCTIONS}"',
        "-c", "project_doc_max_bytes=0",
    ]  # fmt: skip
    for feat in DISABLED_FEATURES:
        args += ["-c", f"features.{feat}=false"]
    return args


def audit_events(events_path: Path) -> dict:
    """Summarize a turn's event stream: item types, tool items, usage, thread id, errors."""
    event_types: collections.Counter = collections.Counter()
    item_types: collections.Counter = collections.Counter()
    tool_items: collections.Counter = collections.Counter()
    usage, thread_id, errors, turn_completed = {}, None, [], False
    for line in events_path.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("type", "")
        event_types[kind] += 1
        if kind == "thread.started":
            thread_id = ev.get("thread_id")
        elif kind == "turn.completed":
            turn_completed, usage = True, ev.get("usage", {})
        elif kind in ("error", "turn.failed"):
            errors.append(json.dumps(ev)[:800])
        elif kind == "item.completed":
            item_type = ev.get("item", {}).get("type", "?")
            item_types[item_type] += 1
            if item_type not in NON_TOOL_ITEMS:
                tool_items[item_type] += 1
    return {
        "event_types": dict(event_types),
        "item_types": dict(item_types),
        "tool_items": dict(tool_items),
        "tool_free": not tool_items,
        "usage": usage,
        "thread_id": thread_id,
        "errors": errors,
        "turn_completed": turn_completed,
    }


def run_turn(
    prompt: str,
    turn_dir: Path,
    *,
    thread_id: str | None = None,
    effort: str = "xhigh",
    model: str = MODEL,
    cwd: Path | None = None,
    timeout_s: int = 3 * 3600,
) -> TurnResult:
    """Send one message to Astra (new thread, or resume `thread_id`) and record everything in `turn_dir`."""
    turn_dir.mkdir(parents=True, exist_ok=True)
    cwd = cwd or (turn_dir.parent / "_empty_cwd")
    cwd.mkdir(parents=True, exist_ok=True)
    answer_path = turn_dir / "answer.md"
    answer_path.unlink(missing_ok=True)
    (turn_dir / "prompt.txt").write_text(prompt)

    common = ["--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "--json", "-m", model]
    common += _config_args(effort) + ["-o", str(answer_path)]
    if thread_id is None:
        cmd = [CODEX, "exec", "-s", "read-only", *common, "-"]
    else:
        cmd = [CODEX, "exec", "resume", *common, thread_id, "-"]

    start = time.time()
    with open(turn_dir / "events.jsonl", "w") as out, open(turn_dir / "stderr.log", "w") as err:
        try:
            rc = subprocess.run(
                cmd, input=prompt, text=True, stdout=out, stderr=err, cwd=cwd, timeout=timeout_s, check=False
            ).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    wall = round(time.time() - start, 1)

    audit = audit_events(turn_dir / "events.jsonl")
    answer = answer_path.read_text() if answer_path.exists() else ""
    ok = rc == 0 and audit["turn_completed"] and bool(answer.strip())
    record = {
        "ok": ok, "rc": rc, "wall_s": wall, "model": model, "effort": effort, "audit": audit,
        "cmd": [redact(part) for part in cmd], "thread_in": thread_id, "end": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }  # fmt: skip
    (turn_dir / "run.json").write_text(json.dumps(record, indent=1))
    return TurnResult(
        ok=ok, answer=answer, thread_id=audit["thread_id"] or thread_id, tool_free=audit["tool_free"],
        usage=audit["usage"], wall_s=wall, rc=rc, errors=audit["errors"], item_types=audit["item_types"],
        cmd=cmd,
    )  # fmt: skip


if __name__ == "__main__":
    import sys

    out_root = Path(sys.argv[1])
    probe = (
        "Reply with the single word OK on the first line. On the second line, list the exact names of "
        "every tool or function you are able to call in this session, or the word none. On the third "
        "line, say whether you were given any instructions besides this message (quote them briefly)."
    )
    first = run_turn(probe, out_root / "t001", effort="low")
    print("turn 1:", first.ok, "tool_free:", first.tool_free, "thread:", first.thread_id, first.usage)
    print(first.answer[:3000])
    print("errors:", first.errors[:2])
    if first.ok:
        second = run_turn("What was the first word of your previous reply?", out_root / "t002",
                          thread_id=first.thread_id, effort="low")  # fmt: skip
        print("turn 2 (resume):", second.ok, "tool_free:", second.tool_free, repr(second.answer[:200]))
        print("errors:", second.errors[:2])
