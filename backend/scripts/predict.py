from pathlib import Path
import argparse
import json
from functools import lru_cache

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "backend" / "models"

MODEL_PATH = MODEL_DIR / "geoflare_rf.joblib"
ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
FEATURES_PATH = MODEL_DIR / "feature_names.json"

CLASSIFICATION_THRESHOLD = 0.70


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the trained model and feature metadata once."""

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    with open(
        FEATURES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        feature_names = json.load(file)

    return model, encoder, feature_names


def predict_dataframe(df):
    """Run GeoFlare classification inference."""

    model, encoder, feature_names = (
        load_artifacts()
    )

    missing = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

    X = df[feature_names].copy()

    predictions = model.predict(X)

    probabilities = (
        model.predict_proba(X)
    )

    model_predicted_labels = (
        encoder.inverse_transform(
            predictions
        )
    )

    prediction_confidence = (
        probabilities.max(axis=1)
    )

    classification_status = [
        (
            "CLASSIFIED"
            if confidence >= CLASSIFICATION_THRESHOLD
            else "UNCERTAIN"
        )
        for confidence in prediction_confidence
    ]

    final_labels = [
        (
            label
            if confidence >= CLASSIFICATION_THRESHOLD
            else "uncertain"
        )
        for label, confidence
        in zip(
            model_predicted_labels,
            prediction_confidence,
        )
    ]

    result = df.copy()

    # Raw model output.
    result["model_predicted_class"] = (
        model_predicted_labels
    )

    result["prediction_confidence"] = (
        prediction_confidence
    )

    # Human-facing GeoFlare classification.
    result["predicted_class"] = (
        final_labels
    )

    result["classification_status"] = (
        classification_status
    )

    # Store probability for every class.
    for index, class_name in enumerate(
        encoder.classes_
    ):
        result[
            f"prob_{class_name}"
        ] = probabilities[:, index]

    return result


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run uncertainty-aware "
            "GeoFlare classification inference."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input CSV containing "
            "GeoFlare ML features."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output CSV path."
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: "
            f"{input_path}"
        )

    print(
        "Loading input data..."
    )

    df = pd.read_csv(
        input_path
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        "\nRunning inference..."
    )

    result = predict_dataframe(
        df
    )

    print(
        "\nClassification distribution:"
    )

    print(
        result[
            "predicted_class"
        ].value_counts()
    )

    print(
        "\nClassification status:"
    )

    print(
        result[
            "classification_status"
        ].value_counts()
    )

    print(
        "\nSample predictions:"
    )

    display_columns = [
        column
        for column in [
            "latitude",
            "longitude",
            "frp",
            "confidence_score",
            "persistence_days",
            "model_predicted_class",
            "predicted_class",
            "prediction_confidence",
            "classification_status",
        ]
        if column in result.columns
    ]

    print(
        result[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    if args.output:

        output_path = Path(
            args.output
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
            "\nSaved predictions to:"
        )

        print(
            output_path
        )


if __name__ == "__main__":
    main()