from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub.errors import GatedRepoError
from huggingface_hub import snapshot_download


def ensure_sam2_checkpoint(checkpoint_path: str | Path) -> Path:
    from ultralytics import SAM

    target_path = Path(checkpoint_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return target_path

    # Ultralytics downloads SAM2 weights when initialized by basename.
    SAM(target_path.name)

    downloaded_candidates = [
        Path.cwd() / target_path.name,
        Path.home() / ".cache" / "ultralytics" / target_path.name,
        Path.home() / ".config" / "Ultralytics" / target_path.name,
    ]
    for candidate in downloaded_candidates:
        if candidate.exists():
            if candidate.resolve() != target_path.resolve():
                shutil.move(str(candidate), str(target_path))
            return target_path

    return target_path


def ensure_sam3_model_dir(model_dir: Path, token: str | None = None) -> Path:
    model_dir = Path(model_dir)
    required_files = [
        model_dir / "model.safetensors",
        model_dir / "sam3.pt",
        model_dir / "pytorch_model.bin",
    ]
    if any(path.exists() for path in required_files):
        return model_dir

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id="facebook/sam3",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            "facebook/sam3 gated repository erisimi gerekiyor. "
            "HF_TOKEN tanimla ve hesapla modele erisim iste."
        ) from exc
    return model_dir
