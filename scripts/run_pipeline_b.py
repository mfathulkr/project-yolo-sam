from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pool_segmentation_compare.config import load_config, resolve_path
from pool_segmentation_compare.pipelines.pipeline_b import run_sam3_hosted_pipeline, run_sam3_local_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text-prompt SAM3 segmentation on the evaluation split.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)

    split = config["dataset"]["eval_split"]
    sam3_cfg = config["sam3"]
    backend = sam3_cfg.get("backend", "hosted")
    images_dir = resolve_path(config["paths"]["prepared_dataset_dir"]) / split / "images"
    output_dir = resolve_path(config["paths"]["pipeline_b_output_dir"])
    prompt = args.prompt or sam3_cfg["prompt"]

    if backend == "hosted":
        api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise SystemExit("ROBOFLOW_API_KEY tanimli degil. .env dosyasina ekle veya --api-key ver.")

        run_sam3_hosted_pipeline(
            images_dir=images_dir,
            output_dir=output_dir,
            api_key=api_key,
            endpoint=sam3_cfg["endpoint"],
            prompt=prompt,
            model_id=sam3_cfg["model_id"],
            output_prob_thresh=sam3_cfg["output_prob_thresh"],
        )
        return

    if backend == "local":
        run_sam3_local_pipeline(
            images_dir=images_dir,
            output_dir=output_dir,
            model_dir=resolve_path(sam3_cfg["local_model_dir"]),
            prompt=prompt,
            device=sam3_cfg["device"],
            torch_dtype=sam3_cfg["torch_dtype"],
            output_prob_thresh=sam3_cfg["output_prob_thresh"],
            mask_threshold=sam3_cfg["mask_threshold"],
            hf_token=os.getenv("HF_TOKEN"),
        )
        return

    raise SystemExit(f"Bilinmeyen SAM3 backend: {backend}")


if __name__ == "__main__":
    main()
