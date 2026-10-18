from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "geoflare_labeled_dataset.csv"
)


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    print("Loading labeled GeoFlare dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Records: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    labeled = df[
        df["target"] != "uncertain"
    ].copy()

    print(f"Labeled records: {len(labeled):,}")

    # ---------------------------------------------------------
    # 1. Class distribution
    # ---------------------------------------------------------

    print_section("CLASS DISTRIBUTION")

    class_counts = (
        labeled["target"]
        .value_counts()
    )

    class_percentages = (
        labeled["target"]
        .value_counts(normalize=True)
        * 100
    )

    for label in class_counts.index:
        print(
            f"{label:28s} "
            f"{class_counts[label]:7,} "
            f"({class_percentages[label]:6.2f}%)"
        )

    # ---------------------------------------------------------
    # 2. Numeric feature summary
    # ---------------------------------------------------------

    print_section("NUMERIC FEATURES BY CLASS")

    numeric_features = [
        "frp",
        "confidence_score",
        "persistence_days",
        "detections_at_location",
        "max_frp_location",
        "distance_to_industry_km",
        "distance_to_power_km",
        "distance_to_oil_gas_km",
        "distance_to_road_km",
        "distance_to_settlement_km",
    ]

    existing_features = [
        column
        for column in numeric_features
        if column in labeled.columns
    ]

    summary = (
        labeled
        .groupby("target")[existing_features]
        .agg(["mean", "median"])
        .round(3)
    )

    print(summary.to_string())

    # ---------------------------------------------------------
    # 3. Land-cover distribution
    # ---------------------------------------------------------

    print_section("LAND-COVER BY CLASS")

    landcover_table = pd.crosstab(
        labeled["target"],
        labeled["landcover_name"],
        normalize="index",
    ) * 100

    print(
        landcover_table
        .round(2)
        .to_string()
    )

    # ---------------------------------------------------------
    # 4. OSM context
    # ---------------------------------------------------------

    print_section("OSM CONTEXT BY CLASS")

    osm_features = [
        "near_industry_5km",
        "near_power_10km",
        "near_oil_gas_10km",
        "near_road_1km",
        "near_settlement_5km",
        "industrial_context",
    ]

    existing_osm = [
        column
        for column in osm_features
        if column in labeled.columns
    ]

    osm_summary = (
        labeled
        .groupby("target")[existing_osm]
        .mean()
        * 100
    )

    print(
        osm_summary
        .round(2)
        .to_string()
    )

    # ---------------------------------------------------------
    # 5. Label confidence
    # ---------------------------------------------------------

    print_section("LABEL CONFIDENCE")

    confidence_summary = (
        labeled
        .groupby("target")["label_confidence"]
        .agg(
            ["count", "mean", "median", "min", "max"]
        )
        .round(3)
    )

    print(
        confidence_summary
        .to_string()
    )

    # ---------------------------------------------------------
    # 6. Label reasons
    # ---------------------------------------------------------

    print_section("LABEL REASONS")

    reasons = (
        labeled
        .groupby(
            ["target", "label_reason"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            ["target", "count"],
            ascending=[True, False],
        )
    )

    for target in labeled["target"].unique():
        print()
        print(f"[{target}]")

        subset = reasons[
            reasons["target"] == target
        ]

        for _, row in subset.iterrows():
            print(
                f"  {row['count']:,} - "
                f"{row['label_reason']}"
            )

    # ---------------------------------------------------------
    # 7. High-FRP analysis
    # ---------------------------------------------------------

    print_section("FRP DISTRIBUTION")

    print(
        labeled
        .groupby("target")["frp"]
        .describe()
        .round(3)
        .to_string()
    )

    # ---------------------------------------------------------
    # 8. Potentially suspicious labels
    # ---------------------------------------------------------

    print_section("SUSPICIOUS LABEL CHECKS")

    checks = {
        "gas_flare_without_oil_gas_context": (
            (labeled["target"] == "gas_flare")
            & (labeled["near_oil_gas_10km"] == 0)
        ),
        "industrial_fire_without_industrial_context": (
            (labeled["target"] == "industrial_fire")
            & (labeled["industrial_context"] == 0)
        ),
        "agricultural_burning_not_cropland": (
            (labeled["target"] == "agricultural_burning")
            & (labeled["landcover_class"] != 40)
        ),
        "natural_fire_without_vegetation": (
            (labeled["target"] == "natural_fire")
            & (~labeled["landcover_class"].isin([10, 20, 30]))
        ),
        "persistent_source_low_persistence": (
            (labeled["target"] == "persistent_thermal_source")
            & (labeled["persistence_days"] < 7)
        ),
    }

    for name, mask in checks.items():
        count = int(mask.sum())

        print(
            f"{name:42s}: {count:,}"
        )

    # ---------------------------------------------------------
    # 9. Duplicate coordinates / locations
    # ---------------------------------------------------------

    print_section("LOCATION CHECK")

    location_columns = [
        "location_id",
        "latitude",
        "longitude",
    ]

    existing_location = [
        column
        for column in location_columns
        if column in labeled.columns
    ]

    if "location_id" in existing_location:
        location_counts = (
            labeled
            .groupby("location_id")
            .size()
        )

        print(
            f"Unique labeled locations: "
            f"{location_counts.shape[0]:,}"
        )

        print(
            f"Locations with multiple labels: "
            f"{(location_counts > 1).sum():,}"
        )

        conflicting_locations = (
            labeled
            .groupby("location_id")["target"]
            .nunique()
        )

        print(
            f"Locations with conflicting classes: "
            f"{(conflicting_locations > 1).sum():,}"
        )

    # ---------------------------------------------------------
    # 10. Final recommendation
    # ---------------------------------------------------------

    print_section("AUDIT COMPLETE")

    print(
        "Use the results above to decide whether the weak-label "
        "rules need adjustment before model training."
    )


if __name__ == "__main__":
    main()