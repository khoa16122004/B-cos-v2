import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

repo_root = Path(__file__).resolve().parents[1]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from main_nsga import main as run_main_nsga


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run main_nsga.py on sampled ImageNet images (typically 1000 images from 1000 classes) "
            "created by evaluate_imagenet1k_correct_and_sample.py."
        )
    )
    parser.add_argument(
        "--samples_json",
        type=Path,
        required=True,
        help="Path to sampled_1000_from_1000_classes_by_model.json",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional single model to run. If omitted, run all models in samples_json.",
    )
    parser.add_argument(
        "--main_nsga_script",
        type=Path,
        default=Path("main_nsga.py"),
        help="Path to main_nsga.py",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("results/nsga_attack_sampled_1000"),
        help="Root directory for per-image attack outputs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--population_size", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--mutation_sigma", type=float, default=0.35)
    parser.add_argument("--crossover_alpha_min", type=float, default=0.2)
    parser.add_argument("--crossover_alpha_max", type=float, default=0.8)
    parser.add_argument(
        "--score_mode",
        type=str,
        default="mean",
        choices=["descriptor_1", "descriptor_2", "mean"],
    )
    parser.add_argument(
        "--score_objective",
        type=str,
        default="min",
        choices=["min", "max"],
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional cap for debugging.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip sample if output summary.json already exists.",
    )
    return parser.parse_args()


def load_samples(samples_json: Path) -> Dict[str, List[Dict[str, object]]]:
    with open(samples_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    models = payload.get("models", {})
    if not isinstance(models, dict) or len(models) == 0:
        raise ValueError("samples_json does not contain any model entries under 'models'.")

    result: Dict[str, List[Dict[str, object]]] = {}
    for model_name, model_data in models.items():
        sampled = model_data.get("sampled_records", [])
        if not isinstance(sampled, list):
            raise ValueError(f"Invalid sampled_records format for model={model_name}")
        result[model_name] = sampled
    return result


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def build_nsga_args(
    model_name: str,
    image_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> argparse.Namespace:
    return argparse.Namespace(
        model_name=model_name,
        image_path=image_path,
        output_dir=output_dir,
        seed=args.seed,
        num_iterations=args.num_iterations,
        population_size=args.population_size,
        epsilon=args.epsilon,
        mutation_sigma=args.mutation_sigma,
        crossover_alpha_min=args.crossover_alpha_min,
        crossover_alpha_max=args.crossover_alpha_max,
        score_mode=args.score_mode,
        score_objective=args.score_objective,
    )


def main(args: argparse.Namespace) -> None:
    if not args.samples_json.exists():
        raise FileNotFoundError(f"samples_json not found: {args.samples_json}")
    if not args.main_nsga_script.exists():
        raise FileNotFoundError(f"main_nsga_script not found: {args.main_nsga_script}")

    samples_by_model = load_samples(args.samples_json)
    if args.model_name is not None:
        if args.model_name not in samples_by_model:
            raise ValueError(f"model_name={args.model_name} not found in samples_json")
        samples_by_model = {args.model_name: samples_by_model[args.model_name]}

    args.output_root.mkdir(parents=True, exist_ok=True)

    report = {
        "samples_json": str(args.samples_json),
        "main_nsga_script": str(args.main_nsga_script),
        "output_root": str(args.output_root),
        "runs": [],
        "num_total": 0,
        "num_success": 0,
        "num_failed": 0,
        "num_skipped": 0,
    }

    for model_name, records in samples_by_model.items():
        selected_records = records if args.max_images is None else records[: args.max_images]

        for idx, record in enumerate(selected_records):
            image_path = Path(str(record["image_path"]))
            class_key = str(record.get("key", "unknown"))
            image_tag = sanitize_name(image_path.stem)

            out_dir = args.output_root / sanitize_name(model_name) / sanitize_name(class_key) / f"{idx:04d}_{image_tag}"
            summary_path = out_dir / "summary.json"

            if args.skip_existing and summary_path.exists():
                report["num_skipped"] += 1
                report["num_total"] += 1
                report["runs"].append(
                    {
                        "model_name": model_name,
                        "image_path": str(image_path),
                        "class_key": class_key,
                        "output_dir": str(out_dir),
                        "status": "skipped",
                        "reason": "summary.json exists",
                    }
                )
                continue

            out_dir.mkdir(parents=True, exist_ok=True)

            nsga_args = build_nsga_args(
                model_name=model_name,
                image_path=image_path,
                output_dir=out_dir,
                args=args,
            )

            error_message = ""
            try:
                run_main_nsga(nsga_args)
                status = "success"
                return_code = 0
            except Exception as exc:
                status = "failed"
                return_code = 1
                error_message = f"{type(exc).__name__}: {exc}"

            report["num_total"] += 1
            if status == "success":
                report["num_success"] += 1
            else:
                report["num_failed"] += 1

            report["runs"].append(
                {
                    "model_name": model_name,
                    "image_path": str(image_path),
                    "class_key": class_key,
                    "pred": int(record.get("pred", -1)),
                    "output_dir": str(out_dir),
                    "status": status,
                    "return_code": int(return_code),
                    "error": error_message,
                }
            )

            print(
                f"[{status}] model={model_name} class={class_key} image={image_path.name} "
                f"-> {out_dir}"
            )

    report_path = args.output_root / "run_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved run report: {report_path}")
    print(
        f"Total={report['num_total']} success={report['num_success']} "
        f"failed={report['num_failed']} skipped={report['num_skipped']}"
    )


if __name__ == "__main__":
    main(parse_args())
