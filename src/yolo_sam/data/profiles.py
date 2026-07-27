from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetProfile:
    profile_id: str
    display_name: str
    categories: tuple[str, ...]
    reference_type: str
    annotation_format: str
    expected_instance_keys: frozenset[str]


SAMRS_SOTA_PROFILE = DatasetProfile(
    profile_id="samrs_sota",
    display_name="SAMRS SOTA",
    categories=(
        "large-vehicle",
        "swimming-pool",
        "helicopter",
        "bridge",
        "plane",
        "ship",
        "soccer-ball-field",
        "basketball-court",
        "ground-track-field",
        "small-vehicle",
        "baseball-diamond",
        "tennis-court",
        "roundabout",
        "storage-tank",
        "harbor",
        "container-crane",
        "airport",
        "helipad",
    ),
    reference_type="pseudo_sam1",
    annotation_format="samrs_pickle_instances",
    expected_instance_keys=frozenset({"mask", "label", "category", "rhbox"}),
)

ISAID_PROFILE = DatasetProfile(
    profile_id="isaid",
    display_name="iSAID",
    categories=(
        "ship",
        "storage_tank",
        "baseball_diamond",
        "tennis_court",
        "basketball_court",
        "Ground_Track_Field",
        "Bridge",
        "Large_Vehicle",
        "Small_Vehicle",
        "Helicopter",
        "Swimming_pool",
        "Roundabout",
        "Soccer_ball_field",
        "plane",
        "Harbor",
    ),
    reference_type="human",
    annotation_format="coco_instance_segmentation",
    expected_instance_keys=frozenset(
        {"id", "image_id", "category_id", "segmentation", "bbox"}
    ),
)


DATASET_PROFILES = {
    SAMRS_SOTA_PROFILE.profile_id: SAMRS_SOTA_PROFILE,
    ISAID_PROFILE.profile_id: ISAID_PROFILE,
}


def normalize_category_name(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def get_dataset_profile(profile_id: str) -> DatasetProfile:
    try:
        return DATASET_PROFILES[profile_id]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_PROFILES))
        raise ValueError(f"Unknown dataset profile {profile_id!r}. Available profiles: {available}") from exc
