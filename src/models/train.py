import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import config

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import mlflow
import mlflow.sklearn
from scipy.stats import ks_2samp

features = [
    "trips_last_7d",
    "trips_last_90d",
    "trips_lifetime",
    "monetary_total",
    "monetary_avg",
    "monetary_last_30d",
    "avg_trip_duration",
    "avg_fare",
    "avg_surge",
    "tip_rate",
    "account_age_days",
    "was_referred",
    "avg_rating_given",
    "loyalty_encoded",
    "city_encoded",
]

target = "churned"


def prepare_train_test(train, test):
    X_train = train[features]
    y_train = train[target]
    X_test  = test[features]
    y_test  = test[target]
    return X_train, y_train, X_test, y_test


def ks_features_diagnostic(train):
    print(f"\n{'Feature':<30}  {'KS':>6}  {'p-value':>8}  Signal")
    print("-" * 60)
    for col in features:
        stats, p = ks_2samp(
            train[train[target] == 0][col].dropna(),
            train[train[target] == 1][col].dropna()
        )
        signal = "STRONG" if p < 0.01 else ("weak" if p < 0.05 else "none")
        print(f"{col:<30}  {stats:>6.3f}  {p:>8.4f}  {signal}")


def train_logistic_regression(X_train, y_train, X_test, y_test):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000))
    ])
    pipe.fit(X_train, y_train)
    proba_test = pipe.predict_proba(X_test)[:, 1]
    roc_auc  = roc_auc_score(y_test, proba_test)
    pr_score = average_precision_score(y_test, proba_test)
    print(f"\n[Logistic Regression]  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_score:.4f}")
    with mlflow.start_run(run_name="logistic_regression", nested=True):
        mlflow.log_params({
            "model": "logistic_regression",
            "class_weight": "balanced",
            "max_iter": 2000,
            "n_features": len(features),
        })
        mlflow.log_metrics({"roc_auc": roc_auc, "pr_auc": pr_score})
        mlflow.sklearn.log_model(pipe, artifact_path="lr_model")
    return roc_auc


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    proba_gb    = gb.predict_proba(X_test)[:, 1]
    gb_roc_auc  = roc_auc_score(y_test, proba_gb)
    gb_pr_score = average_precision_score(y_test, proba_gb)
    print(f"[Gradient Boosting]    ROC-AUC={gb_roc_auc:.4f}  PR-AUC={gb_pr_score:.4f}")
    with mlflow.start_run(run_name="gradient_boosting", nested=True):
        mlflow.log_params({
            "model": "gradient_boosting",
            "n_estimators": gb.n_estimators,
            "learning_rate": gb.learning_rate,
            "max_depth": gb.max_depth,
            "subsample": gb.subsample,
            "min_samples_leaf": gb.min_samples_leaf,
            "n_features": len(features),
        })
        mlflow.log_metrics({"roc_auc": gb_roc_auc, "pr_auc": gb_pr_score})
        mlflow.sklearn.log_model(gb, artifact_path="gb_model")
    return gb_roc_auc


def train(train=None, test=None):
    if train is None:
        train = pd.read_csv(os.path.join(config.DATA_DIR, "train.csv"))
    if test is None:
        test = pd.read_csv(os.path.join(config.DATA_DIR, "test.csv"))

    print(f"Train size : {len(train):,}  |  Test size: {len(test):,}")
    print(f"Churn rate - train: {train[target].mean():.2%}  test: {test[target].mean():.2%}")

    X_train, y_train, X_test, y_test = prepare_train_test(train, test)

    print("\n-- KS Feature Diagnostic --------------------------------------")
    ks_features_diagnostic(train)

    mlflow.set_experiment(experiment_name="Ridewise-churn-prediction")

    print("\n-- Model Training ---------------------------------------------")
    with mlflow.start_run(run_name="churn_training"):
        lr_auc = train_logistic_regression(X_train, y_train, X_test, y_test)
        gb_auc = train_gradient_boosting(X_train, y_train, X_test, y_test)

    winner = "Gradient Boosting" if gb_auc > lr_auc else "Logistic Regression"
    print(f"\nBest model: {winner}")
    print("Training complete. To view MLflow results run: mlflow ui")


if __name__ == "__main__":
    train()
