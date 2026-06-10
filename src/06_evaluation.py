# -*- coding: utf-8 -*-
"""
06_evaluation.py — Évaluation complète & Analyse de robustesse
==============================================================
Compare baseline vs XGBoost et analyse le comportement du modèle
face aux situations difficiles :
  - Pics de demande imprévus
  - Jours fériés
  - Conditions météo extrêmes

Usage
-----
    python src/06_evaluation.py

Outputs
-------
    data/processed/eda_figures/robustness_*.png
    data/processed/evaluation_report.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from utils import evaluate_all, mae, smape, rmse

PROCESSED_DIR = "data/processed"
FIGURES_DIR = f"{PROCESSED_DIR}/eda_figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES RÉSULTATS
# ─────────────────────────────────────────────────────────────

def load_results():
    """Charge toutes les prédictions disponibles."""
    features = pd.read_parquet(f"{PROCESSED_DIR}/features.parquet")
    features["datetime"] = pd.to_datetime(features["datetime"])

    baseline = pd.read_parquet(f"{PROCESSED_DIR}/baseline_predictions.parquet")
    baseline["datetime"] = pd.to_datetime(baseline["datetime"])

    xgb_preds = pd.read_parquet(f"{PROCESSED_DIR}/xgb_predictions.parquet")
    xgb_preds["datetime"] = pd.to_datetime(xgb_preds["datetime"])

    # Merge pour avoir météo + jours fériés sur le jeu de test
    test_df = xgb_preds.merge(
        features[["datetime", "temperature", "precipitation", "is_rain", "is_snow",
                  "is_holiday", "is_holiday_eve", "is_extreme_temp", "is_weekend"]],
        on="datetime", how="left"
    )
    test_df = test_df.merge(
        baseline[["datetime", "pred_naive", "pred_wwa"]],
        on="datetime", how="left"
    )
    return test_df


# ─────────────────────────────────────────────────────────────
# TABLEAU COMPARATIF GLOBAL
# ─────────────────────────────────────────────────────────────

def print_comparison_table(df):
    """Tableau récapitulatif de tous les modèles."""
    y_true = df["total_rides"].values

    results = []
    for name, pred_col in [
        ("Naïf saisonnier (7j)", "pred_naive"),
        ("Moyenne pondérée (4sem)", "pred_wwa"),
        ("XGBoost + Features", "pred_xgb"),
    ]:
        if pred_col not in df.columns:
            continue
        pred = df[pred_col].values
        mask = ~np.isnan(pred)
        results.append({
            "Modèle": name,
            "MAE": mae(y_true[mask], pred[mask]),
            "RMSE": rmse(y_true[mask], pred[mask]),
            "sMAPE (%)": smape(y_true[mask], pred[mask]),
        })

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print("  TABLEAU COMPARATIF — MODÈLES (jeu de test)")
    print("=" * 65)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("=" * 65)

    # Amélioration relative XGBoost vs meilleure baseline
    best_baseline_mae = results_df[results_df["Modèle"] != "XGBoost + Features"]["MAE"].min()
    xgb_mae = results_df[results_df["Modèle"] == "XGBoost + Features"]["MAE"].values[0]
    gain = 100 * (best_baseline_mae - xgb_mae) / best_baseline_mae
    print(f"\n  🎯 Gain XGBoost vs meilleure baseline : -{gain:.1f}% MAE")

    results_df.to_csv(f"{PROCESSED_DIR}/evaluation_report.csv", index=False)
    return results_df


# ─────────────────────────────────────────────────────────────
# ANALYSE DE ROBUSTESSE
# ─────────────────────────────────────────────────────────────

def analyse_peak_demand(df):
    """
    Comportement sur les pics de demande.
    Les pics = heures dans le top 10% de la distribution.
    Problème opérationnel : sous-estimation des pics = stations vides.
    """
    threshold = df["total_rides"].quantile(0.90)
    peak_mask = df["total_rides"] >= threshold

    print(f"\n  📈 PICS DE DEMANDE (> {threshold:.0f} trajets/h, top 10%)")
    print(f"     {peak_mask.sum()} heures de pic sur {len(df)}")

    for name, col in [("Naïf saisonnier", "pred_naive"),
                       ("XGBoost", "pred_xgb")]:
        if col not in df.columns:
            continue
        sub = df[peak_mask]
        bias = (sub[col] - sub["total_rides"]).mean()
        m = mae(sub["total_rides"].values, sub[col].values)
        direction = "sous-estime" if bias < 0 else "sur-estime"
        print(f"     [{name}] MAE={m:.1f} | Biais={bias:.1f} ({direction})")

    # Graphique scatter : réel vs prédit sur les pics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, col, color) in zip(axes, [
        ("Naïf saisonnier", "pred_naive", "steelblue"),
        ("XGBoost", "pred_xgb", "crimson"),
    ]):
        if col not in df.columns:
            continue
        x = df.loc[peak_mask, "total_rides"]
        y = df.loc[peak_mask, col]
        ax.scatter(x, y, alpha=0.3, color=color, s=10)
        lim = max(x.max(), y.max())
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="Parfait")
        ax.set_xlabel("Demande réelle")
        ax.set_ylabel("Demande prédite")
        ax.set_title(f"{name} — Pics (top 10%)", fontweight="bold")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/robustness_peaks.png", dpi=150)
    plt.close()
    print("  ✓ Graphique pics sauvegardé")


def analyse_holidays(df):
    """Comportement sur les jours fériés."""
    if "is_holiday" not in df.columns or df["is_holiday"].isna().all():
        print("\n  ⚠️  Données jours fériés non disponibles")
        return

    print("\n  🗓️  JOURS FÉRIÉS")
    for period, mask_fn in [
        ("Jours fériés", lambda d: d["is_holiday"] == 1),
        ("Jours normaux", lambda d: d["is_holiday"] == 0),
    ]:
        mask = mask_fn(df)
        n = mask.sum()
        if n == 0:
            continue
        print(f"\n     {period} ({n} heures) :")
        sub = df[mask]
        for name, col in [("Naïf", "pred_naive"), ("XGBoost", "pred_xgb")]:
            if col not in sub.columns:
                continue
            m = mae(sub["total_rides"].values, sub[col].values)
            bias = (sub[col] - sub["total_rides"]).mean()
            print(f"       [{name}] MAE={m:.1f} | Biais={bias:.1f}")


def analyse_weather(df):
    """Comportement selon les conditions météo."""
    if "is_rain" not in df.columns or df["is_rain"].isna().all():
        print("\n  ⚠️  Données météo non disponibles")
        return

    print("\n  🌦️  CONDITIONS MÉTÉOROLOGIQUES")

    conditions = {
        "Clair": df["is_rain"].fillna(0) + df["is_snow"].fillna(0) == 0,
        "Pluie": df["is_rain"].fillna(0) == 1,
        "Neige": df["is_snow"].fillna(0) == 1,
        "Temp. extrême": df["is_extreme_temp"].fillna(0) == 1,
    }

    rows = []
    for cond_name, mask in conditions.items():
        n = mask.sum()
        if n < 10:
            continue
        sub = df[mask]
        row = {"Condition": cond_name, "N heures": n,
               "Demande moy": sub["total_rides"].mean()}
        for name, col in [("MAE Naïf", "pred_naive"), ("MAE XGBoost", "pred_xgb")]:
            if col.split("_", 1)[1] not in df.columns:
                row[name] = np.nan
            else:
                row[name] = mae(sub["total_rides"].values,
                                sub[col.split("_", 1)[1]].values)
        rows.append(row)

    cond_df = pd.DataFrame(rows)
    print(cond_df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # Graphique : MAE par condition
    if len(rows) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(cond_df))
        w = 0.35
        bars1 = ax.bar(x - w/2, cond_df.get("MAE Naïf", 0), w,
                       label="Naïf saisonnier", color="steelblue", alpha=0.8)
        bars2 = ax.bar(x + w/2, cond_df.get("MAE XGBoost", 0), w,
                       label="XGBoost", color="crimson", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_df["Condition"])
        ax.set_ylabel("MAE (trajets/h)")
        ax.set_title("Erreur par condition météo — Analyse de robustesse",
                     fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/robustness_weather.png", dpi=150)
        plt.close()
        print("  ✓ Graphique météo sauvegardé")


def plot_error_distribution(df):
    """Distribution des résidus — recherche de biais systématiques."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (name, col, color) in zip(axes, [
        ("Naïf saisonnier", "pred_naive", "steelblue"),
        ("XGBoost", "pred_xgb", "crimson"),
    ]):
        if col not in df.columns:
            continue
        residuals = df["total_rides"] - df[col]
        residuals = residuals.dropna()

        ax.hist(residuals, bins=80, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", lw=1.5, linestyle="--")
        ax.axvline(residuals.mean(), color="red", lw=1.5,
                   label=f"Biais moyen : {residuals.mean():.1f}")
        ax.set_xlabel("Résidu (réel − prédit)")
        ax.set_ylabel("Fréquence")
        ax.set_title(f"Distribution des résidus — {name}", fontweight="bold")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/residuals_distribution.png", dpi=150)
    plt.close()
    print("  ✓ Distribution des résidus sauvegardée")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n🔍 Évaluation complète & Analyse de robustesse")
    print("=" * 55)

    # Chargement
    print("\n  Chargement des prédictions...")
    try:
        df = load_results()
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        print("  → Lancer d'abord les scripts 04 et 05.")
        return

    print(f"  {len(df):,} heures de test chargées")

    # Comparaison globale
    print_comparison_table(df)

    # Analyses de robustesse
    print("\n" + "=" * 55)
    print("  ANALYSES DE ROBUSTESSE")
    print("=" * 55)
    analyse_peak_demand(df)
    analyse_holidays(df)
    analyse_weather(df)
    plot_error_distribution(df)

    print("\n✅ Évaluation terminée. Figures dans :", FIGURES_DIR)
    print("\n  📋 Synthèse des limites identifiées :")
    print("     • Pics liés aux événements imprévus → sous-estimés")
    print("     • Jours fériés atypiques → comportement de semaine vs week-end non capturé")
    print("     • Météo : données prévisionnelles en prod ≠ données observées (bruit)")
    print("     • Drift saisonnier : re-entraîner mensuellement recommandé")


if __name__ == "__main__":
    main()

