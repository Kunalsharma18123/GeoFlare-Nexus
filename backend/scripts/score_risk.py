from pathlib import Path
import argparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


CLASS_WEIGHTS = {
    "industrial_fire": 1.00,
    "natural_fire": 0.90,
    "agricultural_burning": 0.55,
    "persistent_thermal_source": 0.85,
    "uncertain": 0.70,
}


def normalize(value, low, high):
    """Scale a value to the range 0-1."""

    if pd.isna(value):
        return 0.0

    if high <= low:
        return 0.0

    return float(
        np.clip(
            (value - low) / (high - low),
            0.0,
            1.0,
        )
    )


def proximity_score(distance_km, threshold_km):
    """Return a higher score for closer infrastructure."""

    if pd.isna(distance_km):
        return 0.0

    if distance_km >= threshold_km:
        return 0.0

    return float(
        np.clip(
            1.0 - distance_km / threshold_km,
            0.0,
            1.0,
        )
    )


def calculate_risk(row):
    """
    Calculate an explainable GeoFlare risk score.

    Risk score represents event prioritization.
    It is not a physical fire-danger probability.
    """

    reasons = []

    # ---------------------------------------------------------
    # 1. Thermal intensity
    # ---------------------------------------------------------

    frp = float(row["frp"])

    frp_score = normalize(
        frp,
        1.0,
        50.0,
    )

    if frp >= 50:
        reasons.append(
            "Extremely high thermal intensity"
        )
    elif frp >= 25:
        reasons.append(
            "Very high thermal intensity"
        )
    elif frp >= 15:
        reasons.append(
            "High thermal intensity"
        )

    # ---------------------------------------------------------
    # 2. Satellite confidence
    # ---------------------------------------------------------

    satellite_confidence = float(
        row["confidence_score"]
    )

    if satellite_confidence >= 0.80:
        reasons.append(
            "High satellite detection confidence"
        )
    elif satellite_confidence >= 0.66:
        reasons.append(
            "Moderate satellite detection confidence"
        )

    # ---------------------------------------------------------
    # 3. Recent persistence
    # ---------------------------------------------------------

    persistence_days = float(
        row["persistence_days"]
    )

    persistence_score = normalize(
        persistence_days,
        1.0,
        7.0,
    )

    if persistence_days >= 7:
        reasons.append(
            "Long-duration thermal activity"
        )
    elif persistence_days >= 3:
        reasons.append(
            "Persistent thermal activity"
        )

    # ---------------------------------------------------------
    # 4. Repeated detections
    # ---------------------------------------------------------

    detections = float(
        row["detections_at_location"]
    )

    detection_score = normalize(
        detections,
        1.0,
        20.0,
    )

    if detections >= 20:
        reasons.append(
            "Repeated detections at the same location"
        )
    elif detections >= 5:
        reasons.append(
            "Multiple detections at the same location"
        )

    # ---------------------------------------------------------
    # 5. OSM industrial context
    # ---------------------------------------------------------

    industry_score = proximity_score(
        row["distance_to_industry_km"],
        10.0,
    )

    power_score = proximity_score(
        row["distance_to_power_km"],
        10.0,
    )

    oil_gas_score = proximity_score(
        row["distance_to_oil_gas_km"],
        20.0,
    )

    if row["near_industry_5km"]:
        reasons.append(
            "Industrial infrastructure within 5 km"
        )

    if row["near_power_10km"]:
        reasons.append(
            "Power infrastructure within 10 km"
        )

    if row["near_oil_gas_10km"]:
        reasons.append(
            "Oil/gas infrastructure within 10 km"
        )

    # ---------------------------------------------------------
    # 6. Settlement context
    # ---------------------------------------------------------

    settlement_score = proximity_score(
        row["distance_to_settlement_km"],
        5.0,
    )

    if row["near_settlement_5km"]:
        reasons.append(
            "Settlement within 5 km"
        )

    # ---------------------------------------------------------
    # 7. Land-cover context
    # ---------------------------------------------------------

    agriculture_context = float(
        row["agriculture_context"]
    )

    vegetation_context = float(
        row["vegetation_context"]
    )

    builtup_context = float(
        row["builtup_context"]
    )

    # ---------------------------------------------------------
    # 8. ML classification
    # ---------------------------------------------------------

    predicted_class = str(
        row["predicted_class"]
    )

    model_class = str(
        row.get(
            "model_predicted_class",
            predicted_class,
        )
    )

    prediction_confidence = float(
        row["prediction_confidence"]
    )

    is_uncertain = (
        predicted_class == "uncertain"
    )

    if is_uncertain:

        reasons.append(
            "ML classification is uncertain"
        )

        reasons.append(
            f"Top model hypothesis: {model_class}"
        )

    elif predicted_class == "industrial_fire":

        reasons.append(
            "ML classification: industrial fire"
        )

    elif predicted_class == "natural_fire":

        reasons.append(
            "ML classification: natural fire"
        )

    elif predicted_class == "agricultural_burning":

        reasons.append(
            "ML classification: agricultural burning"
        )

    elif predicted_class == "persistent_thermal_source":

        reasons.append(
            "ML classification: persistent thermal source"
        )

    # ---------------------------------------------------------
    # 9. Base risk
    # ---------------------------------------------------------

    base_score = (
        25.0 * frp_score
        + 15.0 * satellite_confidence
        + 15.0 * persistence_score
        + 10.0 * detection_score
        + 10.0 * industry_score
        + 7.0 * power_score
        + 5.0 * oil_gas_score
        + 5.0 * settlement_score
        + 8.0 * prediction_confidence
    )

    # ---------------------------------------------------------
    # 10. Classification context
    # ---------------------------------------------------------

    class_weight = CLASS_WEIGHTS.get(
        predicted_class,
        0.70,
    )

    context_multiplier = 1.0

    if predicted_class == "industrial_fire":

        context_multiplier += (
            0.15
            * max(
                industry_score,
                power_score,
                oil_gas_score,
            )
        )

    elif predicted_class == "persistent_thermal_source":

        context_multiplier += (
            0.10
            * persistence_score
        )

    elif predicted_class == "natural_fire":

        context_multiplier += (
            0.05
            * vegetation_context
        )

    elif predicted_class == "agricultural_burning":

        context_multiplier -= (
            0.10
            * agriculture_context
        )

    elif predicted_class == "uncertain":

        # Do not heavily penalize an event simply because
        # the classifier is uncertain. The physical/context
        # signals still contribute to risk.
        context_multiplier = 1.0

    if builtup_context:
        context_multiplier += 0.05

    risk_score = (
        base_score
        * class_weight
        * context_multiplier
    )

    risk_score = float(
        np.clip(
            risk_score,
            0.0,
            100.0,
        )
    )

    # ---------------------------------------------------------
    # 11. Risk level
    # ---------------------------------------------------------

    if risk_score >= 75:

        risk_level = "CRITICAL"

    elif risk_score >= 50:

        risk_level = "HIGH"

    elif risk_score >= 25:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    # ---------------------------------------------------------
    # 12. Uncertainty flag
    # ---------------------------------------------------------

    if is_uncertain:

        decision_status = (
            "REVIEW_RECOMMENDED"
        )

    else:

        decision_status = (
            "CLASSIFICATION_AVAILABLE"
        )

    return pd.Series(
        {
            "risk_score": round(
                risk_score,
                2,
            ),
            "risk_level": risk_level,
            "decision_status": decision_status,
            "risk_reasons": "; ".join(
                reasons
            ),
        }
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Calculate uncertainty-aware "
            "GeoFlare risk scores."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "CSV containing GeoFlare "
            "predictions and features."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output CSV containing "
            "GeoFlare risk scores."
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{input_path}"
        )

    print(
        "Loading predictions..."
    )

    df = pd.read_csv(
        input_path
    )

    print(
        f"Rows: {len(df):,}"
    )

    required_columns = [
        "frp",
        "confidence_score",
        "persistence_days",
        "detections_at_location",
        "distance_to_industry_km",
        "distance_to_power_km",
        "distance_to_oil_gas_km",
        "distance_to_settlement_km",
        "near_industry_5km",
        "near_power_10km",
        "near_oil_gas_10km",
        "near_settlement_5km",
        "agriculture_context",
        "vegetation_context",
        "builtup_context",
        "predicted_class",
        "prediction_confidence",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    print(
        "Calculating risk scores..."
    )

    risk_results = df.apply(
        calculate_risk,
        axis=1,
    )

    result = pd.concat(
        [
            df,
            risk_results,
        ],
        axis=1,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nClassification status:"
    )

    if "classification_status" in result:

        print(
            result[
                "classification_status"
            ].value_counts()
        )

    print(
        "\nRisk distribution:"
    )

    print(
        result[
            "risk_level"
        ].value_counts()
    )

    print(
        "\nHighest-risk events:"
    )

    display_columns = [
        "latitude",
        "longitude",
        "frp",
        "predicted_class",
        "prediction_confidence",
        "risk_score",
        "risk_level",
        "decision_status",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in result.columns
    ]

    print(
        result.nlargest(
            10,
            "risk_score",
        )[
            available_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved risk-scored dataset to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()