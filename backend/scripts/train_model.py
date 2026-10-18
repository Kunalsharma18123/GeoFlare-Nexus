from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "ml"
    / "train.csv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "ml"
    / "test.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
)

MODEL_FILE = MODEL_DIR / "geoflare_rf.joblib"
ENCODER_FILE = MODEL_DIR / "label_encoder.joblib"
FEATURE_FILE = MODEL_DIR / "feature_names.json"
CM_FILE = MODEL_DIR / "confusion_matrix.png"
IMPORTANCE_FILE = MODEL_DIR / "feature_importance.csv"


FEATURES = [
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


def main():
    print("Loading training and testing datasets...")

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    print(f"Training records: {len(train):,}")
    print(f"Testing records:  {len(test):,}")

    # ---------------------------------------------------------
    # Validate features
    # ---------------------------------------------------------

    missing_train = [
        column
        for column in FEATURES
        if column not in train.columns
    ]

    missing_test = [
        column
        for column in FEATURES
        if column not in test.columns
    ]

    if missing_train:
        raise ValueError(
            f"Missing training features: {missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Missing testing features: {missing_test}"
        )

    # ---------------------------------------------------------
    # Prepare X and y
    # ---------------------------------------------------------

    X_train = train[FEATURES].copy()
    X_test = test[FEATURES].copy()

    y_train = train[TARGET].copy()
    y_test = test[TARGET].copy()

    # ---------------------------------------------------------
    # Encode target labels
    # ---------------------------------------------------------

    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(
        y_train
    )

    y_test_encoded = encoder.transform(
        y_test
    )

    print()
    print("Classes:")

    for index, label in enumerate(
        encoder.classes_
    ):
        print(f"  {index}: {label}")

    # ---------------------------------------------------------
    # Train Random Forest
    # ---------------------------------------------------------

    print()
    print("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train_encoded,
    )

    print("Training completed.")

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    print()
    print("Evaluating model...")

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test_encoded,
        predictions,
    )

    print()
    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print()
    print("Classification report:")

    report = classification_report(
        y_test_encoded,
        predictions,
        target_names=encoder.classes_,
        digits=4,
    )

    print(report)

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_test_encoded,
        predictions,
    )

    print()
    print("Confusion matrix:")

    print(cm)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoder.classes_,
    )

    display.plot(
        xticks_rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        CM_FILE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        IMPORTANCE_FILE,
        index=False,
    )

    print()
    print("Top 15 features:")

    print(
        importance.head(15)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Save model artifacts
    # ---------------------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE,
    )

    joblib.dump(
        encoder,
        ENCODER_FILE,
    )

    with FEATURE_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            FEATURES,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print()
    print("Model artifacts saved:")

    print(
        f"  Model: {MODEL_FILE}"
    )

    print(
        f"  Encoder: {ENCODER_FILE}"
    )

    print(
        f"  Features: {FEATURE_FILE}"
    )

    print(
        f"  Confusion matrix: {CM_FILE}"
    )

    print(
        f"  Feature importance: {IMPORTANCE_FILE}"
    )

    print()
    print("GeoFlare baseline training complete.")


if __name__ == "__main__":
    main()