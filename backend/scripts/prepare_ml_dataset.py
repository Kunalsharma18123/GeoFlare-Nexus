from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "geoflare_labeled_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "ml"
)

TRAIN_FILE = OUTPUT_DIR / "train.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"
CLEAN_FILE = OUTPUT_DIR / "clean_labeled.csv"


ML_FEATURES = [
    "frp",
    "confidence_score",
    "latitude",
    "longitude",
    "hour",
    "minute",
    "day_of_year",
    "day_of_week",
    "month",
    "is_weekend",
    "is_day",
    "persistence_days",
    "detections_at_location",
    "mean_frp_location",
    "max_frp_location",
    "min_frp_location",
    "frp_std_location",
    "distance_to_industry_km",
    "distance_to_power_km",
    "distance_to_oil_gas_km",
    "distance_to_road_km",
    "distance_to_settlement_km",
    "near_industry_5km",
    "near_power_10km",
    "near_oil_gas_10km",
    "near_road_1km",
    "near_settlement_5km",
    "industrial_context",
    "settlement_context",
    "road_context",
    "vegetation_context",
    "agriculture_context",
    "builtup_context",
    "water_context",
    "wetland_context",
]


TARGET = "target"
GROUP = "location_id"


def main():
    print("Loading GeoFlare labeled dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total records: {len(df):,}")

    # ---------------------------------------------------------
    # 1. Keep only weakly labeled records
    # ---------------------------------------------------------

    df = df[
        df[TARGET] != "uncertain"
    ].copy()

    print(
        f"Labeled records: {len(df):,}"
    )

    # ---------------------------------------------------------
    # 2. Find locations with conflicting labels
    # ---------------------------------------------------------

    label_counts = (
        df.groupby(GROUP)[TARGET]
        .nunique()
    )

    conflicting_locations = (
        label_counts[
            label_counts > 1
        ]
        .index
    )

    print(
        f"Conflicting locations: "
        f"{len(conflicting_locations):,}"
    )

    # Remove conflicting locations completely.
    df = df[
        ~df[GROUP].isin(
            conflicting_locations
        )
    ].copy()

    print(
        f"Records after conflict removal: "
        f"{len(df):,}"
    )

    # ---------------------------------------------------------
    # 3. Remove gas flare from supervised ML
    # ---------------------------------------------------------

    gas_flare_count = (
        df[TARGET] == "gas_flare"
    ).sum()

    print(
        f"Gas flare records excluded from "
        f"supervised training: "
        f"{gas_flare_count:,}"
    )

    df = df[
        df[TARGET] != "gas_flare"
    ].copy()

    # ---------------------------------------------------------
    # 4. Keep only required ML features
    # ---------------------------------------------------------

    required_columns = (
        ML_FEATURES
        + [TARGET, GROUP]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing ML columns: {missing}"
        )

    ml_df = df[
        required_columns
    ].copy()

    # ---------------------------------------------------------
    # 5. Remove rows with missing feature values
    # ---------------------------------------------------------

    before = len(ml_df)

    ml_df = ml_df.dropna(
        subset=ML_FEATURES
    ).copy()

    removed = before - len(ml_df)

    print(
        f"Rows removed because of missing "
        f"features: {removed:,}"
    )

    # ---------------------------------------------------------
    # 6. Show final class distribution
    # ---------------------------------------------------------

    print()
    print("Final class distribution:")

    distribution = (
        ml_df[TARGET]
        .value_counts()
    )

    for label, count in distribution.items():
        percentage = (
            count / len(ml_df)
        ) * 100

        print(
            f"  {label:28s}"
            f"{count:7,}"
            f" ({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------
    # 7. Spatial group split
    # ---------------------------------------------------------

    print()
    print("Creating location-grouped train/test split...")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(
            ml_df,
            groups=ml_df[GROUP],
        )
    )

    train_df = ml_df.iloc[
        train_indices
    ].copy()

    test_df = ml_df.iloc[
        test_indices
    ].copy()

    # ---------------------------------------------------------
    # 8. Verify location leakage
    # ---------------------------------------------------------

    train_locations = set(
        train_df[GROUP]
    )

    test_locations = set(
        test_df[GROUP]
    )

    overlap = (
        train_locations
        & test_locations
    )

    print(
        f"Train records: {len(train_df):,}"
    )

    print(
        f"Test records:  {len(test_df):,}"
    )

    print(
        f"Train locations: "
        f"{len(train_locations):,}"
    )

    print(
        f"Test locations:  "
        f"{len(test_locations):,}"
    )

    print(
        f"Location overlap: {len(overlap):,}"
    )

    if overlap:
        raise RuntimeError(
            "Location leakage detected!"
        )

    # ---------------------------------------------------------
    # 9. Save datasets
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ml_df.to_csv(
        CLEAN_FILE,
        index=False,
    )

    train_df.to_csv(
        TRAIN_FILE,
        index=False,
    )

    test_df.to_csv(
        TEST_FILE,
        index=False,
    )

    print()
    print("ML dataset preparation completed.")

    print(
        f"Clean dataset: {CLEAN_FILE}"
    )

    print(
        f"Training dataset: {TRAIN_FILE}"
    )

    print(
        f"Testing dataset: {TEST_FILE}"
    )

    print()
    print("Training class distribution:")

    print(
        train_df[TARGET]
        .value_counts()
        .to_string()
    )

    print()
    print("Testing class distribution:")

    print(
        test_df[TARGET]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()