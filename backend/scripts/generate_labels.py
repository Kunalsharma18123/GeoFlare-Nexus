from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "firms_historical_osm_landcover_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "geoflare_labeled_dataset.csv"
)


def main():
    print("Loading enriched GeoFlare dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Records: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    required_columns = {
        "frp",
        "confidence_score",
        "persistence_days",
        "detections_at_location",
        "max_frp_location",
        "landcover_class",
        "landcover_name",
        "distance_to_industry_km",
        "distance_to_power_km",
        "distance_to_oil_gas_km",
        "near_industry_5km",
        "near_power_10km",
        "near_oil_gas_10km",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    print()
    print("Calculating dynamic thresholds...")

    high_frp_threshold = df["frp"].quantile(0.90)
    very_high_frp_threshold = df["frp"].quantile(0.95)

    print(
        f"90th percentile FRP: "
        f"{high_frp_threshold:.2f}"
    )

    print(
        f"95th percentile FRP: "
        f"{very_high_frp_threshold:.2f}"
    )

    labels = np.full(
        len(df),
        "uncertain",
        dtype=object,
    )

    label_confidence = np.zeros(
        len(df),
        dtype=float,
    )

    label_reason = np.full(
        len(df),
        "No strong contextual signal",
        dtype=object,
    )

    label_priority = np.zeros(
        len(df),
        dtype=int,
    )

    def assign(
        mask,
        label,
        confidence,
        reason,
        priority,
    ):
        eligible = (
            mask
            & (priority > label_priority)
        )

        labels[eligible] = label
        label_confidence[eligible] = confidence
        label_reason[eligible] = reason
        label_priority[eligible] = priority

    frp = df["frp"]
    persistence = df["persistence_days"]
    detections = df["detections_at_location"]
    max_frp = df["max_frp_location"]

    confidence = df["confidence_score"]

    landcover = df["landcover_class"]

    near_industry = (
        df["near_industry_5km"] == 1
    )

    near_power = (
        df["near_power_10km"] == 1
    )

    near_oil_gas = (
        df["near_oil_gas_10km"] == 1
    )

    distance_oil_gas = (
        df["distance_to_oil_gas_km"]
    )

    print()
    print("Generating weak labels...")

    # ---------------------------------------------------------
    # 1. GAS FLARE
    # ---------------------------------------------------------

    gas_flare = (
        near_oil_gas
        & (persistence >= 3)
        & (detections >= 3)
        & (max_frp >= high_frp_threshold)
    )

    assign(
        gas_flare,
        "gas_flare",
        0.88,
        "Repeated high-FRP detections near mapped oil/gas infrastructure",
        5,
    )

    # ---------------------------------------------------------
    # 2. PERSISTENT THERMAL SOURCE
    # ---------------------------------------------------------

    persistent_source = (
        (persistence >= 7)
        & (detections >= 7)
        & (max_frp >= high_frp_threshold)
    )

    assign(
        persistent_source,
        "persistent_thermal_source",
        0.86,
        "Long-duration repeated thermal activity at the same location",
        4,
    )

    # ---------------------------------------------------------
    # 3. INDUSTRIAL FIRE
    # ---------------------------------------------------------

    industrial_fire = (
        (
            near_industry
            | near_power
        )
        & (frp >= very_high_frp_threshold)
        & (persistence <= 3)
        & (confidence >= 0.66)
    )

    assign(
        industrial_fire,
        "industrial_fire",
        0.78,
        "High-FRP short-duration detection near industrial or power infrastructure",
        3,
    )

    # ---------------------------------------------------------
    # 4. AGRICULTURAL BURNING
    # ---------------------------------------------------------

    agricultural_burning = (
        (landcover == 40)
        & (frp >= high_frp_threshold)
        & (persistence <= 3)
        & (detections >= 1)
    )

    assign(
        agricultural_burning,
        "agricultural_burning",
        0.76,
        "High-FRP short-duration detection over cropland",
        2,
    )

    # ---------------------------------------------------------
    # 5. NATURAL / FOREST FIRE
    # ---------------------------------------------------------

    natural_fire = (
        landcover.isin([10, 20, 30])
        & (frp >= high_frp_threshold)
        & (persistence <= 5)
        & (confidence >= 0.66)
        & ~near_industry
        & ~near_power
        & ~near_oil_gas
    )

    assign(
        natural_fire,
        "natural_fire",
        0.74,
        "High-confidence thermal activity in vegetated land without strong industrial context",
        1,
    )

    # ---------------------------------------------------------
    # Label outputs
    # ---------------------------------------------------------

    df["target"] = labels
    df["label_confidence"] = label_confidence
    df["label_reason"] = label_reason

    df["is_labeled"] = (
        df["target"] != "uncertain"
    ).astype("int8")

    df["label_priority"] = label_priority

    print()
    print("Label generation completed.")

    print()
    print("Label distribution:")

    distribution = (
        df["target"]
        .value_counts()
    )

    for label, count in distribution.items():
        percentage = (
            count / len(df)
        ) * 100

        print(
            f"  {label}: "
            f"{count:,} ({percentage:.2f}%)"
        )

    print()
    print(
        "Labeled records: "
        f"{df['is_labeled'].sum():,}"
    )

    print(
        "Uncertain records: "
        f"{(df['is_labeled'] == 0).sum():,}"
    )

    print()
    print("Label confidence:")

    labeled = df[
        df["is_labeled"] == 1
    ]

    if not labeled.empty:
        print(
            f"  Mean: "
            f"{labeled['label_confidence'].mean():.3f}"
        )

        print(
            f"  Median: "
            f"{labeled['label_confidence'].median():.3f}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(f"Output: {OUTPUT_FILE}")
    print(f"Final columns: {len(df.columns)}")


if __name__ == "__main__":
    main()