# -*- coding: utf-8 -*-
"""
04_baseline_model.py — Modèle Baseline
=======================================
Implémente deux baselines :
  1. Naïf saisonnier : demande = valeur même heure, il y a 7 jours
  2. Moyenne glissante pondérée sur 4 semaines précédentes

Ces baselines servent de référence pour juger l'apport des modèles avancés.
Un bon modèle ML doit les battre SIGNIFICATIVEMENT pour justifier sa complexité.

Usage
-----
    python src/04_baseline_model.py

Outputs
-------
    data/processed/baseline_predictions.parquet
    data/processed/eda_figures/baseline_analysis.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from utils import evaluate_all, temporal_split

PROCESSED_DIR = "data/processed"
os.makedirs(f"{PROCESSED_DIR}/eda_figures", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# BASELINE 1 : NAÏF SAISONNIER (7 jours)
# ─────────────────────────────────────────────────────────────

class SeasonalNaive:
    """
    Prédiction = valeur observée il y a S périodes.
    Pour S=168 (7 jours × 24h) et horizon=48h :
      prédiction(t) = observation(t - 168h)
    
    C'est la baseline de référence en séries temporelles saisonnières.
    Simple, interprétable, et souvent difficile à battre sur données régulières.
    """
    def __init__(self, seasonality: int = 168, horizon: int = 48):
        self.seasonality = seasonality
        self.horizon = horizon

    def predict(self, y: pd.Series, n_ahead: int) -> np.ndarray:
        """
        Prédit les n_ahead prochaines valeurs.
        Utilise uniquement les données passées.
        """
        preds = []
        for i in range(n_ahead):
            idx = len(y) - n_ahead + i
            look_back = idx - self.seasonality + self.horizon
            if look_back >= 0:
                preds.append(y.iloc[look_back])
            else:
                preds.append(np.nan)
        return np.array(preds)

    def predict_series(self, y_series: pd.Series) -> pd.Series:
        """Prédit toute la série par décalage simple."""
        return y_series.shift(self.seasonality)


# ─────────────────────────────────────────────────────────────
# BASELINE 2 : MOYENNE GLISSANTE MULTI-SEMAINES
# ─────────────────────────────────────────────────────────────

class WeightedWeeklyAverage:
    """
    Prédiction = moyenne pondérée des 4 mêmes heures des 4 semaines passées.
    Poids décroissants : semaine récente = poids fort.
    
    Avantage sur le naïf : lisse le bruit aléatoire d'une semaine atypique.
    """
    def __init__(self, n_weeks: int = 4, horizon: int = 48):
        self.n_weeks = n_weeks
        self.horizon = horizon
        self.weights = np.array([1, 2, 3, 4][:n_weeks], dtype=float)
        self.weights = self.weights / self.weights.sum()  # normalisation

    def predict_series(self, y_series: pd.Series) -> pd.Series:
        """Calcul vectorisé des prédictions pour toute la série."""
        preds = pd.Series(index=y_series.index, dtype=float)
        h = 168  # 1 semaine en heures

        for i in range(self.n_weeks * h, len(y_series)):
            lags = [y_series.iloc[i - k * h] for k in range(1, self.n_weeks + 1)]
            preds.iloc[i] = np.dot(lags, self.weights[::-1])

        return preds


# ─────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────

def plot_baseline_results(test_df, pred_naive, pred_weighted):
    """Graphique comparant les deux baselines sur le jeu de test."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Prendre 2 semaines de test pour lisibilité
    n_plot = min(14 * 24, len(test_df))
    dt = test_df["datetime"].iloc[:n_plot]
    y_true = test_df["total_rides"].iloc[:n_plot]

    for ax, pred, label, color in [
        (axes[0], pred_naive[:n_plot], "Naïf saisonnier (7j)", "steelblue"),
        (axes[1], pred_weighted[:n_plot], "Moyenne pondérée (4 semaines)", "darkorange"),
    ]:
        ax.plot(dt, y_true, color="black", lw=1, alpha=0.7, label="Réel")
        ax.plot(dt, pred, color=color, lw=1.5, alpha=0.8, label=label, linestyle="--")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
        ax.legend()
        ax.set_ylabel("Trajets/heure")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_title("Évaluation des baselines — Jeu de test", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PROCESSED_DIR}/eda_figures/baseline_analysis.png", dpi=150)
    plt.close()
    print("  ✓ Graphique baseline sauvegardé")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n📐 Modèle Baseline")
    print("=" * 50)

    # Chargement
    df = pd.read_parquet(f"{PROCESSED_DIR}/features.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Split temporel strict
    print("\n  Split temporel :")
    train, val, test = temporal_split(df, "datetime",
                                      train_end="2025-06-30",
                                      val_end="2025-9-30")
    
    # Réinitialiser les index après le split
    train = train.reset_index(drop=True)
    val   = val.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    y_train = train["total_rides"]
    y_val = val["total_rides"]
    y_test = test["total_rides"]

    # ─── Baseline 1 : Naïf saisonnier ───
    print("\n  Baseline 1 : Naïf saisonnier (168h)")
    naive = SeasonalNaive(seasonality=168, horizon=48)

    # La prédiction sur val/test utilise TOUT ce qui précède
    # Reconstruire la série sur df complet réinitialisé
    df_reset = df.reset_index(drop=True)
    full_series = df_reset["total_rides"]
    pred_naive_all = naive.predict_series(full_series)

    # Aligner par datetime plutôt que par position
    val_dates  = val["datetime"]
    test_dates = test["datetime"]

    pred_naive_val  = pred_naive_all[df_reset["datetime"].isin(val_dates)].values
    pred_naive_test = pred_naive_all[df_reset["datetime"].isin(test_dates)].values

    metrics_naive_val = evaluate_all(y_val.values, pred_naive_val,
                                     y_train=y_train, label="Naïf saisonnier — Val")
    metrics_naive_test = evaluate_all(y_test.values, pred_naive_test,
                                      y_train=y_train, label="Naïf saisonnier — Test")

    # ─── Baseline 2 : Moyenne pondérée ───
    print("\n  Baseline 2 : Moyenne pondérée 4 semaines")
    wwa = WeightedWeeklyAverage(n_weeks=4, horizon=48)

    pred_wwa_all = wwa.predict_series(full_series)
    pred_wwa_val  = pred_wwa_all[df_reset["datetime"].isin(val_dates)].values
    pred_wwa_test = pred_wwa_all[df_reset["datetime"].isin(test_dates)].values

    metrics_wwa_val = evaluate_all(y_val.values, pred_wwa_val,
                                   y_train=y_train, label="Moyenne pondérée — Val")
    metrics_wwa_test = evaluate_all(y_test.values, pred_wwa_test,
                                    y_train=y_train, label="Moyenne pondérée — Test")

    print(f"Train : {len(train)} | Val : {len(val)} | Test : {len(test)}")
    print(f"pred_naive_val  : {len(pred_naive_val)}")
    print(f"pred_naive_test : {len(pred_naive_test)}")
    
    # ─── Visualisation ───
    print("\n  📊 Génération du graphique...")
    plot_baseline_results(test, pred_naive_test, pred_wwa_test)

    # ─── Sauvegarde des prédictions ───
    preds_df = test[["datetime", "total_rides"]].copy()
    preds_df["pred_naive"] = pred_naive_test
    preds_df["pred_wwa"] = pred_wwa_test
    preds_df.to_parquet(f"{PROCESSED_DIR}/baseline_predictions.parquet", index=False)

    # ─── Sauvegarde des métriques ───
    summary = {
        "naive_val": metrics_naive_val,
        "naive_test": metrics_naive_test,
        "wwa_val": metrics_wwa_val,
        "wwa_test": metrics_wwa_test,
    }
    joblib.dump(summary, f"{PROCESSED_DIR}/baseline_metrics.pkl")

    print(f"\n✅ Baseline terminée.")
    print(f"   Best baseline (MAE test) : "
          f"Naïf={metrics_naive_test['MAE']:.1f} | "
          f"Moyenne ponderée={metrics_wwa_test['MAE']:.1f}")


if __name__ == "__main__":
    main()
