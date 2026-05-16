import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split


DROP_COLUMNS = [
    "Embryo_ID",
    "Case",
    "SlideID",
    "Biopsy_grade0",
    "Day3_grade0",
    "Day3_grade",
    "Day5_grade0",
    "Weight",
    "Infertility_cause",
    "Years_infertility",
    "Sperm_volume",
    "sperm_grade0",
    "sperm_grade1",
    "sperm_grade2",
    "sperm_grade3",
]


def build_model():
    return RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=4,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=14,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Train_Internal_test.csv")
    parser.add_argument("--out", default=".", help="Repository root output directory")
    args = parser.parse_args()

    root = Path(args.out)
    models_dir = root / "models"
    data_dir = root / "data"
    models_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.data)
    df = df.drop(columns=DROP_COLUMNS)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    imputer = IterativeImputer(random_state=14)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=381
    )

    model = build_model()
    kf = KFold(n_splits=5)
    scores = cross_val_score(model, X_train, y_train, cv=kf)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = {
        "cross_validation_accuracy_mean": float(scores.mean()),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_score)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "decision_threshold": 0.402962,
        "features": list(X.columns),
    }

    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "features": list(X.columns),
            "threshold": 0.402962,
            "medians": X.median(numeric_only=True).to_dict(),
        },
        models_dir / "ploidy_random_forest.joblib",
    )

    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(data_dir / "global_feature_importance.csv", index=False)
    X_train.sample(min(500, len(X_train)), random_state=14).to_csv(
        data_dir / "lime_background.csv", index=False
    )

    with (data_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
