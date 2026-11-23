"""Run one forward pass of a MedQA lab session in its own process.

    python -m wsp.lab_worker <job.json>

The lab's host process starts this for every pass and reads the result file. Because the process exits when the
pass is done, model memory can never pile up across passes (an in-process version grew to 18 GB over 20 passes
on Apple MPS and pushed the machine into swap).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from wsp.lab import MedQALab


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    lab = MedQALab.for_worker(Path(job["run_dir"]), job["model"], job["donor"], job["style"])
    try:
        with torch.no_grad():
            summary, where = lab._dispatch_local(job["request"], job["index"])
        result = {"summary": summary, "where": where}
    except Exception as error:  # noqa: BLE001 - every failure is reported back to the session, uncounted
        result = {"error_type": type(error).__name__, "error": str(error)}
    Path(job["out"]).write_text(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1])
