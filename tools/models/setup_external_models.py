from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = ROOT / "external_models"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd or ROOT, check=True)


def ensure_repo(url: str, path: Path) -> None:
    if (path / ".git").exists():
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        if status:
            print(f"Skipping pull for locally patched repo: {path}")
            return
        run(["git", "pull", "--ff-only"], cwd=path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", url, str(path)])


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Patch anchor not found in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def patch_segearth_ov3(repo: Path) -> None:
    replace_once(
        repo / "sam3" / "model" / "position_encoding.py",
        '            for size in precompute_sizes:\n                tensors = torch.zeros((1, 1) + size, device="cuda")\n',
        '            precompute_device = "cuda" if torch.cuda.is_available() else "cpu"\n'
        "            for size in precompute_sizes:\n"
        "                tensors = torch.zeros((1, 1) + size, device=precompute_device)\n",
    )
    replace_once(
        repo / "sam3" / "model" / "decoder.py",
        '                coords_h, coords_w = self._get_coords(\n                    feat_size, feat_size, device="cuda"\n                )\n',
        "                coords_h, coords_w = self._get_coords(\n"
        "                    feat_size,\n"
        "                    feat_size,\n"
        '                    device="cuda" if torch.cuda.is_available() else "cpu",\n'
        "                )\n",
    )
    replace_once(
        repo / "sam3" / "model" / "geometry_encoders.py",
        "            scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype)\n"
        "            scale = scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)\n",
        "            scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype)\n"
        '            if boxes_xyxy.device.type == "cuda":\n'
        "                scale = scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)\n"
        "            else:\n"
        "                scale = scale.to(device=boxes_xyxy.device)\n",
    )


def patch_remotesam(repo: Path) -> None:
    model_path = repo / "tasks" / "code" / "model.py"
    replace_once(
        model_path,
        "from .RuleBasedCaptioning import single_captioning\n",
        "try:\n"
        "    from .RuleBasedCaptioning import single_captioning\n"
        "except ModuleNotFoundError:\n"
        "    single_captioning = None\n",
    )
    replace_once(
        model_path,
        "def embed_sentences(sentences, max_tokens=20):\n"
        "    # init\n"
        "    bert_model = transformers.BertTokenizer.from_pretrained('bert-base-uncased')\n",
        "_BERT_TOKENIZER = None\n"
        "\n"
        "\n"
        "def embed_sentences(sentences, max_tokens=20):\n"
        "    # init\n"
        "    global _BERT_TOKENIZER\n"
        "    if _BERT_TOKENIZER is None:\n"
        "        _BERT_TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased')\n"
        "    bert_model = _BERT_TOKENIZER\n",
    )
    replace_once(
        model_path,
        "    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'], strict=False)\n",
        "    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=False)\n",
    )
    replace_once(
        model_path,
        "    def captioning(self, *, image, classnames, region_split=4):\n"
        "        temp = self.detection(image=image, classnames=classnames)\n",
        "    def captioning(self, *, image, classnames, region_split=4):\n"
        "        if single_captioning is None:\n"
        '            raise RuntimeError("Captioning dependencies are not installed.")\n'
        "        temp = self.detection(image=image, classnames=classnames)\n",
    )

    files = {
        "mmcv/__init__.py": 'from __future__ import annotations\n\n__version__ = "0.0.local_stub"\n\nfrom .utils import mkdir_or_exist\n',
        "mmcv/fileio/__init__.py": (
            "from __future__ import annotations\n\n"
            "import json\n"
            "from pathlib import Path\n\n\n"
            "class FileClient:\n"
            "    def __init__(self, *args, **kwargs) -> None:\n"
            "        self.args = args\n"
            "        self.kwargs = kwargs\n\n"
            "    def get(self, filename: str) -> bytes:\n"
            "        return Path(filename).read_bytes()\n\n\n"
            "def load(filename: str):\n"
            "    path = Path(filename)\n"
            '    with path.open("r", encoding="utf-8") as handle:\n'
            "        return json.load(handle)\n"
        ),
        "mmcv/parallel/__init__.py": (
            "from __future__ import annotations\n\n"
            "import torch\n\n\n"
            "def is_module_wrapper(module) -> bool:\n"
            "    return isinstance(\n"
            "        module,\n"
            "        (\n"
            "            torch.nn.DataParallel,\n"
            "            torch.nn.parallel.DistributedDataParallel,\n"
            "        ),\n"
            "    )\n"
        ),
        "mmcv/runner/__init__.py": "from __future__ import annotations\n\n\ndef get_dist_info() -> tuple[int, int]:\n    return 0, 1\n",
        "mmcv/utils/__init__.py": (
            "from __future__ import annotations\n\n"
            "from pathlib import Path\n\n\n"
            "def mkdir_or_exist(path: str) -> None:\n"
            "    if path:\n"
            "        Path(path).mkdir(parents=True, exist_ok=True)\n"
        ),
    }
    for relative_path, content in files.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")


def main() -> None:
    segearth = EXTERNAL_ROOT / "SegEarth-OV-3"
    remotesam = EXTERNAL_ROOT / "RemoteSAM"
    ensure_repo("https://github.com/earth-insights/SegEarth-OV-3.git", segearth)
    ensure_repo("https://github.com/1e12Leon/RemoteSAM.git", remotesam)
    patch_segearth_ov3(segearth)
    patch_remotesam(remotesam)
    print("External model repositories are ready.")


if __name__ == "__main__":
    main()
