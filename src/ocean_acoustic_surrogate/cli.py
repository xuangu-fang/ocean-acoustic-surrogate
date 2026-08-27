"""Command-line entry points for the complete experiment workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import MVPConfig, load_campaign
from .dataset import generate_dataset, run_pilot
from .reporting import commit_lightweight_results, write_campaign_summary
from .training import run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocean-acoustic-surrogate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("pilot", help="run Bellhop ray-count convergence audit")
    pilot.add_argument("config", type=Path)
    pilot.add_argument("--samples", type=int, default=8)
    generate = subparsers.add_parser("generate", help="generate or resume the Bellhop dataset")
    generate.add_argument("config", type=Path)
    generate.add_argument("--samples", type=int, default=512)
    campaign = subparsers.add_parser("campaign", help="train every registered experiment")
    campaign.add_argument("config", type=Path)
    campaign.add_argument("campaign", type=Path)
    campaign.add_argument("--samples", type=int, default=512)
    campaign.add_argument("--only", nargs="*", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = MVPConfig.from_yaml(args.config)
    if args.command == "pilot":
        print(run_pilot(config, args.samples))
        return
    if args.command == "generate":
        print(generate_dataset(config, args.samples))
        return
    campaign = load_campaign(args.campaign)
    dataset_path = config.dataset_root / f"n{args.samples}" / "dataset.npz"
    if not dataset_path.exists():
        dataset_path = generate_dataset(config, args.samples)
    experiments = campaign["experiments"]
    if args.only:
        selected = set(args.only)
        experiments = [item for item in experiments if item["id"] in selected]
    run_dirs = []
    failures = []
    for experiment in experiments:
        try:
            run_dir = run_experiment(
                config, dataset_path, experiment, campaign["training_defaults"]
            )
            run_dirs.append(run_dir)
            print(f"completed {experiment['id']}: {run_dir}", flush=True)
        except Exception as exc:  # noqa: BLE001 - campaign must preserve other runs
            failures.append({"experiment_id": experiment["id"], "error": repr(exc)})
            print(f"failed {experiment['id']}: {exc!r}", flush=True)
    if not run_dirs:
        raise RuntimeError(f"all experiments failed: {failures}")
    campaign_root = config.artifact_root / "campaigns"
    summary_path = campaign_root / "latest.json"
    summary = write_campaign_summary(run_dirs, summary_path)
    summary["failures"] = failures
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    commit_lightweight_results(summary)
    print(summary_path)


if __name__ == "__main__":
    main()
