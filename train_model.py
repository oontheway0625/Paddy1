from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "paddydataset.csv"
OUT_PATH = BASE_DIR / "paddy_dashboard_bundle.joblib"
RANDOM_STATE = 42
CLASS_LABELS = ["Low", "Moderate", "High", "Very High"]


def make_preprocessor(numeric_cols, categorical_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])


def clean_feature_name(name):
    return name.replace("num__", "").replace("cat__", "")


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    rows_original = len(df)
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    excluded_features = [
        "Hectares", "Seedrate(in Kg)", "LP_Mainfield(in Tonnes)",
        "Nursery area (Cents)", "LP_nurseryarea(in Tonnes)", "DAP_20days",
        "Weed28D_thiobencarb", "Urea_40Days", "Potassh_50Days",
        "Micronutrients_70Days", "Pest_60Day(in ml)",
    ]
    raw_target = "Paddy yield(in Kg)"
    selected_features = [
        c for c in df.columns
        if c not in excluded_features and c != raw_target and c != "Paddy Yield Category"
    ]

    df["Paddy Yield Category"] = pd.cut(
        df[raw_target],
        bins=[0, 10000, 20000, 30000, 40000],
        labels=CLASS_LABELS,
        right=False,
    )

    X = df[selected_features].copy()
    y = df["Paddy Yield Category"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    model_specs = {
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN (Tuned)": KNeighborsClassifier(metric="euclidean", n_neighbors=31, weights="distance"),
        "Random Forest (Tuned)": RandomForestClassifier(
            class_weight="balanced", max_depth=10, max_features="sqrt",
            min_samples_leaf=1, min_samples_split=5, n_estimators=200,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "ANN (Tuned)": MLPClassifier(
            solver="adam", max_iter=800, random_state=RANDOM_STATE,
            activation="tanh", alpha=0.0001, hidden_layer_sizes=(64, 32),
            learning_rate_init=0.001,
        ),
    }

    models = {
        name: Pipeline([
            ("preprocessor", make_preprocessor(numeric_cols, categorical_cols)),
            ("model", estimator),
        ])
        for name, estimator in model_specs.items()
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    metrics = []
    confusion_matrices = {}
    roc_data = {}
    pr_data = {}
    cv_folds = {}

    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)
        classes = list(model.named_steps["model"].classes_)

        metrics.append({
            "Model": name,
            "Accuracy": float(accuracy_score(y_test, pred)),
            "Precision": float(precision_score(y_test, pred, average="macro", zero_division=0)),
            "Recall": float(recall_score(y_test, pred, average="macro", zero_division=0)),
            "F1 Score": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "AUC": float(roc_auc_score(y_test, prob, labels=classes, multi_class="ovr", average="macro")),
            "Log Loss": float(log_loss(y_test, prob, labels=classes)),
        })
        confusion_matrices[name] = confusion_matrix(y_test, pred, labels=CLASS_LABELS).tolist()

        y_bin = label_binarize(y_test, classes=classes)
        roc_model = {}
        pr_model = {}
        for i, class_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
            precision, recall, _ = precision_recall_curve(y_bin[:, i], prob[:, i])
            roc_model[str(class_name)] = {
                "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))
            }
            pr_model[str(class_name)] = {
                "precision": precision.tolist(), "recall": recall.tolist(),
                "ap": float(average_precision_score(y_bin[:, i], prob[:, i])),
            }
        roc_data[name] = roc_model
        pr_data[name] = pr_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_folds[name] = cross_val_score(
                model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1
            ).tolist()

    rf = models["Random Forest (Tuned)"]
    rf_names = [clean_feature_name(x) for x in rf.named_steps["preprocessor"].get_feature_names_out()]
    rf_importance = sorted(
        [
            {"Feature": feature, "Importance": float(value)}
            for feature, value in zip(rf_names, rf.named_steps["model"].feature_importances_)
        ],
        key=lambda row: row["Importance"], reverse=True,
    )

    lr = models["Logistic Regression (Baseline)"]
    lr_names = [clean_feature_name(x) for x in lr.named_steps["preprocessor"].get_feature_names_out()]
    lr_coef = np.mean(np.abs(lr.named_steps["model"].coef_), axis=0)
    lr_importance = sorted(
        [{"Feature": feature, "Importance": float(value)} for feature, value in zip(lr_names, lr_coef)],
        key=lambda row: row["Importance"], reverse=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_sizes, train_scores, val_scores = learning_curve(
            rf, X_train, y_train, cv=cv, scoring="f1_macro",
            train_sizes=np.linspace(0.2, 1.0, 5), shuffle=True,
            random_state=RANDOM_STATE, n_jobs=1,
        )
    rf_learning_curve = {
        "train_sizes": train_sizes.tolist(),
        "train_mean": train_scores.mean(axis=1).tolist(),
        "train_std": train_scores.std(axis=1).tolist(),
        "val_mean": val_scores.mean(axis=1).tolist(),
        "val_std": val_scores.std(axis=1).tolist(),
    }

    class_counts = (
        df["Paddy Yield Category"].value_counts().reindex(CLASS_LABELS).fillna(0).astype(int).to_dict()
    )
    categorical_options = {
        c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in categorical_cols
    }
    numeric_stats = {}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        numeric_stats[c] = {
            "min": float(s.min()), "max": float(s.max()),
            "median": float(s.median()), "mean": float(s.mean()),
        }

    corr = (
        df[numeric_cols + [raw_target]].corr(numeric_only=True)[raw_target]
        .drop(raw_target).sort_values(key=lambda s: s.abs(), ascending=False)
    )
    corr_top = [{"Feature": name, "Correlation": float(value)} for name, value in corr.items()]

    bundle = {
        "models": models,
        "selected_features": selected_features,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "class_labels": CLASS_LABELS,
        "yield_ranges": {
            "Low": "< 10,000 kg",
            "Moderate": "10,000–19,999 kg",
            "High": "20,000–29,999 kg",
            "Very High": "30,000–39,999 kg",
        },
        "metrics": metrics,
        "confusion_matrices": confusion_matrices,
        "roc_data": roc_data,
        "pr_data": pr_data,
        "cv_folds": cv_folds,
        "rf_feature_importance": rf_importance,
        "lr_feature_importance": lr_importance,
        "rf_learning_curve": rf_learning_curve,
        "class_counts": class_counts,
        "categorical_options": categorical_options,
        "numeric_stats": numeric_stats,
        "corr_top": corr_top,
        "excluded_features": excluded_features,
        "best_params": {
            "Logistic Regression (Baseline)": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
            "KNN (Tuned)": {"metric": "euclidean", "n_neighbors": 31, "weights": "distance"},
            "Random Forest (Tuned)": {
                "class_weight": "balanced", "max_depth": 10, "max_features": "sqrt",
                "min_samples_leaf": 1, "min_samples_split": 5, "n_estimators": 200,
            },
            "ANN (Tuned)": {
                "activation": "tanh", "alpha": 0.0001, "hidden_layer_sizes": (64, 32),
                "learning_rate_init": 0.001, "max_iter": 800,
            },
        },
        "rf_tuning_evidence": [
            {"Candidate": 1, "n_estimators": 200, "max_depth": "10", "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": "balanced", "mean_train_f1": 0.9821, "mean_cv_f1": 0.9523, "std_cv_f1": 0.0090},
            {"Candidate": 2, "n_estimators": 300, "max_depth": "10", "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": "balanced", "mean_train_f1": 0.9824, "mean_cv_f1": 0.9520, "std_cv_f1": 0.0059},
            {"Candidate": 3, "n_estimators": 300, "max_depth": "20", "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": "balanced", "mean_train_f1": 0.9825, "mean_cv_f1": 0.9510, "std_cv_f1": 0.0074},
        ],
        "data_summary": {
            "rows_original": rows_original,
            "duplicates_removed": duplicates_removed,
            "rows_cleaned": len(df),
            "raw_features": 45,
            "selected_features": len(selected_features),
            "prepared_features": 60,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
    }

    joblib.dump(bundle, OUT_PATH, compress=3)
    print(f"Saved: {OUT_PATH}")
    print(pd.DataFrame(metrics).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
