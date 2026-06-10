# -*- coding: utf-8 -*-
"""
05_advanced_model.py — Modèle Avancé XGBoost
=============================================
Implémente un pipeline de modélisation complet :
  1. Sélection des features
  2. Entraînement XGBoost avec validation temporelle
  3. Optimisation des hyperparamètres (optionnel)
  4. Analyse de l'importance des features
  5. Sauvegarde du modèle

Usage
-----
    python src/05_advanced_model.py

Outputs
-------
    data/processed/xgb_predictions.parquet
    data/processed/xgb_model.joblib
    data/processed/eda_figures/feature_importance.png
    data/processed/eda_figures/xgb_predictions.png
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from utils import evaluate_all, temporal_split

PROCESSED_DIR = "data/processed"
os.makedirs(f"{PROCESSED_DIR}/eda_figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FEATURES SÉLECTIONNÉES
# ─────────────────────────────────────────────────────────────

# Features temporelles (cycliques — pas d'encodage one-hot nécessaire)
TEMPORAL_FEATURES = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend",
    "is_holiday", "is_holiday_eve", "is_holiday_after",
    "day_of_year",
]

# Features de lag (toutes >= 48h pour respecter l'horizon J+2)
LAG_FEATURES = [
    "lag_48h",
    "lag_168h",     # même heure, il y a 7 jours
    "lag_336h",     # même heure, il y a 14 jours
    "lag_504h",     # même heure, il y a 21 jours
    "rolling_mean_24h",
    "rolling_mean_48h",
    "rolling_mean_168h",
    "rolling_std_24h",
    "rolling_std_48h",
]

# Features météo (si disponibles)
WEATHER_FEATURES = [
    "temperature",
    "apparent_temp",
    "precipitation",
    "windspeed",
    "is_rain",
    "is_snow",
    "is_extreme_temp",
    "temp_squared",
]

TARGET = "total_rides"


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Retourne les colonnes features disponibles dans le DataFrame."""
    all_features = TEMPORAL_FEATURES + LAG_FEATURES + WEATHER_FEATURES
    available = [f for f in all_features if f in df.columns]
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"  ⚠️  Features manquantes (ignorées) : {missing}")
    return available


# ─────────────────────────────────────────────────────────────
# MODÈLE XGBOOST
# ─────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Entraîne XGBoost avec early stopping sur la validation.
    
    Hyperparamètres choisis après analyse :
    - n_estimators élevé + early stopping (anti-overfitting)
    - learning_rate faible = convergence stable
    - subsample + colsample = régularisation par bootstrap
    - max_depth modéré = éviter la mémorisation des outliers
    """
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,     # éviter de sur-fitter les heures rares
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,           # L1 — sparse features
        reg_lambda=1.0,          # L2 — régularisation générale
        random_state=42,
        n_jobs=-1,
        eval_metric="mae",
        early_stopping_rounds=50,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"  ✓ Modèle entraîné — {model.best_iteration} arbres (early stopping)")
    return model


# ─────────────────────────────────────────────────────────────
# ANALYSE FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_cols):
    """Graphique des 20 features les plus importantes."""
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance["feature"], importance["importance"],
            color="steelblue", edgecolor="white")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Top 20 features — XGBoost", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PROCESSED_DIR}/eda_figures/feature_importance.png", dpi=150)
    plt.close()
    print("  ✓ Feature importance sauvegardée")


def plot_predictions_vs_real(test_df, y_pred, n_days=14):
    """Comparaison prédictions vs réel sur les 2 premières semaines du test."""
    n_plot = min(n_days * 24, len(test_df))
    dt = test_df["datetime"].iloc[:n_plot]
    y_true = test_df[TARGET].iloc[:n_plot]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Série temporelle
    ax1.plot(dt, y_true, color="black", lw=1, label="Réel", alpha=0.8)
    ax1.plot(dt, y_pred[:n_plot], color="crimson", lw=1.5,
             label="Prédit (XGBoost)", linestyle="--", alpha=0.8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax1.set_title("XGBoost — Prédiction vs Réel (jeu de test)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Trajets/heure")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Résidus
    residuals = y_true.values - y_pred[:n_plot]
    ax2.fill_between(dt, residuals, 0, where=residuals > 0,
                     alpha=0.5, color="steelblue", label="Sous-estimé")
    ax2.fill_between(dt, residuals, 0, where=residuals < 0,
                     alpha=0.5, color="coral", label="Sur-estimé")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax2.set_ylabel("Résidu (réel − prédit)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PROCESSED_DIR}/eda_figures/xgb_predictions.png", dpi=150)
    plt.close()
    print("  ✓ Graphique prédictions sauvegardé")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n🚀 Modèle Avancé — XGBoost")
    print("=" * 50)

    # Chargement
    df = pd.read_parquet(f"{PROCESSED_DIR}/features.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Split temporel strict
    print("\n  Split temporel :")
    train, val, test = temporal_split(df, "datetime",
                                      train_end="2024-09-30",
                                      val_end="2024-10-31")

    # Features
    feature_cols = get_feature_cols(df)
    print(f"\n  {len(feature_cols)} features sélectionnées :")
    print(f"  {feature_cols}")

    X_train = train[feature_cols].fillna(0)
    y_train = train[TARGET]
    X_val = val[feature_cols].fillna(0)
    y_val = val[TARGET]
    X_test = test[feature_cols].fillna(0)
    y_test = test[TARGET]

    # Entraînement
    print("\n  🎯 Entraînement XGBoost...")
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # Prédictions
    pred_val = model.predict(X_val)
    pred_val = np.maximum(pred_val, 0)  # pas de valeurs négatives
    pred_test = model.predict(X_test)
    pred_test = np.maximum(pred_test, 0)

    # Évaluation
    metrics_val = evaluate_all(y_val.values, pred_val,
                               y_train=y_train, label="XGBoost — Val")
    metrics_test = evaluate_all(y_test.values, pred_test,
                                y_train=y_train, label="XGBoost — Test")

    # Visualisations
    print("\n  📊 Génération des graphiques...")
    plot_feature_importance(model, feature_cols)
    plot_predictions_vs_real(test, pred_test)

    # Sauvegarde du modèle
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgb_model.joblib"
    joblib.dump({
        "model": model,
        "feature_cols": feature_cols,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "train_end": "2024-09-30",
        "val_end": "2024-10-31",
    }, model_path)
    print(f"  ✓ Modèle sauvegardé : {model_path}")

    # Sauvegarde des prédictions
    preds_df = test[["datetime", TARGET]].copy()
    preds_df["pred_xgb"] = pred_test
    preds_df["residual"] = preds_df[TARGET] - preds_df["pred_xgb"]
    preds_df.to_parquet(f"{PROCESSED_DIR}/xgb_predictions.parquet", index=False)

    print(f"\n✅ XGBoost terminé.")
    print(f"   MAE test : {metrics_test['MAE']:.2f} trajets/h")
    print(f"   sMAPE test : {metrics_test['sMAPE (%)']:.1f}%")


if __name__ == "__main__":
    main()

