from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIFT_REPORT = ROOT / "reports" / "monitoring" / "drift_report.json"
DEFAULT_PLAN_OUTPUT = ROOT / "reports" / "monitoring" / "retraining_plan.json"


def run_command(command: list[str], execute: bool) -> dict[str, object]:
    if not execute:
        return {"command": command, "executed": False, "returncode": None}
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return {"command": command, "executed": True, "returncode": completed.returncode}


def should_retrain(report: dict[str, object], force: bool) -> tuple[bool, str]:
    if force:
        return True, "forced by operator"
    status = report.get("status")
    if status == "alert":
        return True, "drift monitor status is alert"
    if status == "watch":
        return False, "drift monitor status is watch; collect review feedback before retraining"
    return False, "drift monitor status is ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate a gated retraining workflow.")
    parser.add_argument("--drift-report", type=Path, default=DEFAULT_DRIFT_REPORT)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--execute-training", action="store_true", help="Actually run training and reports.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.drift_report.exists():
        raise FileNotFoundError(
            f"Drift report not found: {args.drift_report}. Run scripts/drift_monitor.py first."
        )

    report = json.loads(args.drift_report.read_text(encoding="utf-8"))
    retrain, reason = should_retrain(report, args.force)
    commands = []
    if retrain:
        commands = [
            [sys.executable, "scripts/train_model.py", "--epochs", str(args.epochs), "--batch-size", str(args.batch_size)],
            [sys.executable, "scripts/evaluate_threshold.py"],
            [sys.executable, "scripts/calibration_analysis.py"],
            [sys.executable, "scripts/confidence_intervals.py"],
            [sys.executable, "scripts/register_model.py", "--stage", "candidate"],
        ]

    results = [run_command(command, args.execute_training) for command in commands]
    plan = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "drift_status": report.get("status"),
        "retraining_recommended": retrain,
        "reason": reason,
        "execute_training": args.execute_training,
        "commands": results,
        "approval_gate": (
            "Candidate models should be reviewed through metrics, calibration, robustness, "
            "and error analysis before replacing the active model."
        ),
    }
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({"retraining_recommended": retrain, "reason": reason}, indent=2))


if __name__ == "__main__":
    main()
