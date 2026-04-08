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
from pool_segmentation_compare.models.download import ensure_sam3_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download local SAM3 model files.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    config = load_config(parse_args().config)
    target_dir = resolve_path(config["sam3"]["local_model_dir"])
    print(f"Ensuring local SAM3 model files under -> {target_dir}")
    final_dir = ensure_sam3_model_dir(target_dir, token=os.getenv("HF_TOKEN"))
    print(f"SAM3 available at: {final_dir}")


if __name__ == "__main__":
    main()
