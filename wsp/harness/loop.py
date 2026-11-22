"""Turn-by-turn conversation loop between Astra and our tools, in the style of the MIT harness.

Astra has no tools of its own. To act, it replies with exactly one fenced block (```python, or another
block type an experiment defines, such as ```attention or ```forward). The harness runs the block and
sends what it printed back as the next message of the same conversation. A reply without a block is the
final answer. Everything is written to disk in the same layout as the MIT logs:

    <run_dir>/message.txt, state.json, tool_calls.jsonl, final_answer.md, loop.log
    <run_dir>/turns/tNNN/{prompt.txt, answer.md, events.jsonl, run.json, stderr.log}

The loop is resumable: state.json holds the thread id and the next message, so a rerun continues.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from wsp.harness import codex_client

BLOCK_RE = re.compile(r"```([A-Za-z_]+)[ \t]*\n(.*?)```", re.DOTALL)
TOOL_REMINDER = (
    "Your last reply was discarded because it called a Codex tool. No tools are available in this "
    "conversation: reply with text only, as described in the first message.\n\n"
)


@dataclass
class State:
    thread_id: str | None = None
    turn: int = 0
    runs: dict = field(default_factory=dict)  # block type -> how many were run
    nudges: int = 0
    done: bool = False
    next_message: str | None = None
    failures: int = 0
    wall_s: float = 0.0
    handoffs: int = 0
    turns_in_thread: int = 0
    unexplained: int = 0


Handler = Callable[[str, State], str]


@dataclass
class Loop:
    run_dir: Path
    first_message: str
    handlers: dict[str, Handler]
    max_runs: int = 100
    count_as_run: tuple[str, ...] = ("python",)
    effort: str = "xhigh"
    max_turns: int = 400
    retries: int = 4
    retry_wait_s: int = 60
    # Called when Astra replies without a block. Return a message to keep going, or None to accept it.
    on_final: Callable[[str, State], str | None] | None = None
    # Wraps a block's output into the next message.
    format_output: Callable[[str, str, State], str] | None = None
    # Called with (reply, block type) before a block runs. Return a message to send back instead of running it.
    precheck: Callable[[str, str, State], str | None] | None = None
    # Start a fresh conversation every N replies (0 = never). `handoff` returns the text appended to the task.
    handoff_every: int = 0
    handoff: Callable[[State, str], str] | None = None
    # Sent in place of a message the provider keeps refusing (for example after a content-filter refusal).
    undeliverable_notice: str | None = None

    def log(self, text: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(self.run_dir / "loop.log", "a") as f:
            f.write(f"[{stamp}] {self.run_dir.name} {text}\n")

    def _load(self) -> State:
        path = self.run_dir / "state.json"
        return State(**json.loads(path.read_text())) if path.exists() else State()

    def _save(self, state: State) -> None:
        (self.run_dir / "state.json").write_text(json.dumps(asdict(state), indent=1))

    def runs_used(self, state: State) -> int:
        return sum(state.runs.get(kind, 0) for kind in self.count_as_run)

    def _default_format(self, kind: str, output: str, state: State) -> str:
        return f"OUTPUT of run {self.runs_used(state)} of at most {self.max_runs}:\n```\n{output}\n```"

    def _ask(self, message: str, state: State) -> codex_client.TurnResult | None:
        """One turn, retried on empty replies, Codex failures and (never expected) tool use."""
        turn_dir = self.run_dir / "turns" / f"t{state.turn + 1:03d}"
        prompt = message
        for attempt in range(1, self.retries + 1):
            result = codex_client.run_turn(prompt, turn_dir, thread_id=state.thread_id, effort=self.effort)
            state.wall_s += result.wall_s
            if result.ok and result.tool_free:
                return result
            reason = "tool use" if result.ok else f"codex failure (rc {result.rc})"
            self.log(f"turn {state.turn + 1} attempt {attempt}: {reason}; errors {result.errors[:1]}")
            state.failures += 1
            keep = turn_dir.parent / f"{turn_dir.name}_fail{attempt}"
            if turn_dir.exists():
                turn_dir.rename(keep)
            if result.ok and not result.tool_free:
                # The thread now contains the tool call; continue it, but say the reply was discarded.
                state.thread_id = result.thread_id
                prompt = TOOL_REMINDER + message
            time.sleep(self.retry_wait_s)
        return None

    def run(self) -> State:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "message.txt").write_text(self.first_message)
        state = self._load()
        message = state.next_message or self.first_message
        formatter = self.format_output or self._default_format

        while not state.done and state.turn < self.max_turns:
            if (
                self.handoff_every
                and self.handoff
                and state.thread_id
                and state.turns_in_thread >= self.handoff_every
            ):
                # Keep conversations short: restart with the task, the notebook and the last result.
                state.thread_id, state.turns_in_thread = None, 0
                state.handoffs += 1
                message = self.first_message + "\n\n" + self.handoff(state, message)
                self.log(f"turn {state.turn + 1}: handoff {state.handoffs} to a new conversation")
            kind_of_turn = "new thread" if state.thread_id is None else "resume"
            self.log(f"turn {state.turn + 1}: {kind_of_turn} ({len(message):,} chars)")
            result = self._ask(message, state)
            if result is None and self.undeliverable_notice:
                self.log(f"turn {state.turn + 1}: message undeliverable, sending the notice instead")
                result = self._ask(self.undeliverable_notice, state)
            if result is None:
                self.log(f"turn {state.turn + 1}: giving up after {self.retries} attempts")
                self._save(state)
                break
            state.thread_id = result.thread_id
            state.turn += 1
            state.turns_in_thread += 1

            block = next((m for m in BLOCK_RE.finditer(result.answer) if m.group(1) in self.handlers), None)
            if block is None:
                follow_up = self.on_final(result.answer, state) if self.on_final else None
                if follow_up is None:
                    (self.run_dir / "final_answer.md").write_text(result.answer)
                    state.done, state.next_message = True, None
                    self.log(f"turn {state.turn}: final answer ({len(result.answer):,} chars)")
                else:
                    early = self.run_dir / f"early_answer_{state.nudges + 1}.md"
                    early.write_text(result.answer)
                    state.nudges += 1
                    message = state.next_message = follow_up
                    self.log(f"turn {state.turn}: answered early, nudge {state.nudges}")
                self._save(state)
                continue

            kind, code = block.group(1), block.group(2)
            refusal = self.precheck(result.answer, kind, state) if self.precheck else None
            if refusal is not None:
                message = state.next_message = refusal
                self.log(f"turn {state.turn}: {kind} block not run (precheck)")
                self._save(state)
                continue
            if kind in self.count_as_run and self.runs_used(state) >= self.max_runs:
                message = (
                    f"You have used all {self.max_runs} runs, so this block was not run. "
                    "Reply now WITHOUT a code block, with your final answer."
                )
            else:
                start = time.time()
                output = self.handlers[kind](code, state)
                state.runs[kind] = state.runs.get(kind, 0) + 1
                with open(self.run_dir / "tool_calls.jsonl", "a") as f:
                    record = {"t": time.time(), "turn": state.turn, "kind": kind, "run": state.runs[kind],
                              "s": round(time.time() - start, 2), "code": code, "out": output}  # fmt: skip
                    f.write(json.dumps(record) + "\n")
                message = formatter(kind, output, state)
                self.log(f"turn {state.turn}: {kind} run {state.runs[kind]} "
                         f"({len(code):,} chars of code, {len(output):,} chars out), tool-free True")  # fmt: skip
            state.next_message = message
            self._save(state)
        return state
