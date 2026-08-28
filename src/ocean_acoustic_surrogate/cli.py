"""Command-line entry points for the complete experiment workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import MVPConfig, load_campaign


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocean-acoustic-surrogate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("pilot", help="run Bellhop ray-count convergence audit")
    pilot.add_argument("config", type=Path)
    pilot.add_argument("--samples", type=int, default=8)
    generate = subparsers.add_parser("generate", help="generate or resume the Bellhop dataset")
    generate.add_argument("config", type=Path)
    generate.add_argument("--samples", type=int, default=512)
    generate.add_argument(
        "--reuse-prefix-from",
        type=Path,
        default=None,
        help="reuse an identical frozen sample prefix from an earlier dataset root",
    )
    campaign = subparsers.add_parser("campaign", help="train every registered experiment")
    campaign.add_argument("config", type=Path)
    campaign.add_argument("campaign", type=Path)
    campaign.add_argument("--samples", type=int, default=512)
    campaign.add_argument("--only", nargs="*", default=None)
    verify = subparsers.add_parser("verify", help="reload a checkpoint and verify the sealed test")
    verify.add_argument("config", type=Path)
    verify.add_argument("run_dir", type=Path)
    verify.add_argument("--samples", type=int, default=512)
    verify.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    profile = subparsers.add_parser("profile", help="summarize a packaged dataset")
    profile.add_argument("config", type=Path)
    profile.add_argument("--samples", type=int, default=512)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = MVPConfig.from_yaml(args.config)
    if args.command == "pilot":
        from .dataset import run_pilot

        print(run_pilot(config, args.samples))
        return
    if args.command == "generate":
        from .dataset import generate_dataset

        print(generate_dataset(config, args.samples, args.reuse_prefix_from))
        return
    if args.command == "verify":
        from .verification import verify_run

        dataset_path = config.dataset_root / f"n{args.samples}" / "dataset.npz"
        print(verify_run(config, dataset_path, args.run_dir, args.device))
        return
    if args.command == "profile":
        from .reporting import commit_dataset_profile

        dataset_root = config.dataset_root / f"n{args.samples}"
        print(commit_dataset_profile(dataset_root / "dataset.npz", dataset_root / "manifest.json"))
        return
    campaign = load_campaign(args.campaign)
    dataset_path = config.dataset_root / f"n{args.samples}" / "dataset.npz"
    if not dataset_path.exists():
        from .dataset import generate_dataset

        dataset_path = generate_dataset(config, args.samples)
    from .reporting import commit_lightweight_results, write_campaign_summary
    from .training import run_experiment

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
