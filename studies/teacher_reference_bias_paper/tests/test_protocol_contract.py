from __future__ import annotations

import hashlib
import json
from pathlib import Path

from teacher_reference_bias.config import (
    load_dataset_study_config,
    load_matched_study_config,
)
from teacher_reference_bias_multiteacher.paths import DATASETS


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]


def test_protocol_is_single_seed_and_pinned_sam3_pvs() -> None:
    protocol = load_matched_study_config(STUDY_ROOT / "configs" / "protocol.yaml")
    assert protocol.detector_seeds == (42,)
    assert protocol.image_size == 1024
    assert protocol.evaluation["max_per_stratum"] == 128
    assert protocol.segmenter_configs["sam3"]["inference_interface"] == "sam3_tracker_pvs"
    assert protocol.segmenter_configs["sam3"]["mask_threshold"] == 0.0


def test_local_8gb_profile_preserves_model_contract_and_reduces_batches() -> None:
    canonical = load_matched_study_config(STUDY_ROOT / "configs" / "protocol.yaml")
    local = load_matched_study_config(
        STUDY_ROOT / "configs" / "protocol.local_8gb.yaml"
    )
    assert local.detector_seeds == canonical.detector_seeds == (42,)
    assert local.image_size == canonical.image_size == 1024
    assert local.segmenter_configs["sam3"]["checkpoint_sha256"] == (
        canonical.segmenter_configs["sam3"]["checkpoint_sha256"]
    )
    assert local.detector["batch"] < canonical.detector["batch"]
    assert local.segmenter_configs["sam3"]["box_batch_size"] < (
        canonical.segmenter_configs["sam3"]["box_batch_size"]
    )


def test_four_experiment_configs_are_portable() -> None:
    assert set(DATASETS) == {
        "isaid_plane",
        "isaid_small_vehicle",
        "samrs_plane",
        "samrs_small_vehicle",
    }
    for experiment_id, source in DATASETS.items():
        config = load_dataset_study_config(source.root / "config.yaml")
        assert config.experiment_root == source.root
        assert config.prepared_root == source.prepared_root
        assert config.results_root == source.results_root
        assert config.dataset_id == source.dataset_id
        master = load_dataset_study_config(source.root / "master_config.yaml")
        assert config.master_prepared_root == master.prepared_root
        assert master.experiment_root == source.root
        assert master.profile_id == config.profile_id
        assert master.target_category == config.target_category
        assert not (source.prepared_root / "data.yaml").read_text().startswith("path:")


def test_master_provenance_points_to_canonical_hash_verified_manifests() -> None:
    for source in DATASETS.values():
        config = load_dataset_study_config(source.root / "config.yaml")
        provenance = json.loads(
            (source.prepared_root / "master_provenance.json").read_text(
                encoding="utf-8"
            )
        )
        recorded_root = REPO_ROOT / provenance["master_prepared_root"]
        recorded_manifest = REPO_ROOT / provenance["master_content_manifest"]
        assert recorded_root.resolve() == config.master_prepared_root.resolve()
        assert recorded_manifest.resolve() == (
            config.master_prepared_root / "content_manifest.json"
        ).resolve()
        assert hashlib.sha256(recorded_manifest.read_bytes()).hexdigest() == (
            provenance["master_content_manifest_sha256"]
        )
