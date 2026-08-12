from __future__ import annotations

import sys
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (REPO_ROOT / "src", STUDY_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias_multiteacher.paper_assets import generate_assets  # noqa: E402


def main() -> None:
    result = generate_assets(STUDY_ROOT)
    for category in ("figures", "tables", "manifest"):
        for path in result[category]:
            print(path)


if __name__ == "__main__":
    main()
