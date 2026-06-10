# -*- coding: utf-8 -*-

"""
utils.py — Fonctions utilitaires partagées
==========================================
Métriques, helpers temporels, et fonctions communes au pipeline.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# MÉTRIQUES D'ÉVALUATION
# ─────────────────────────────────────────────────────────────

def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def smape(y_true, y_pred, epsilon=1e-8):
    """
    Symmetric Mean Absolute Percentage Error.
    Robuste aux valeurs proches de zéro grâce à epsilon.
    Plage : [0%, 200%]
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    return 100 * np.mean(numerator / denominator)


def mape(y_true, y_pred, epsilon=1e-8):
    """Mean Absolute Percentage Error. Eviter si y_true contient des zéros."""
    return 100 * np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))


def mase(y_true, y_pred, y_train, seasonality=24):
    """
    Mean Absolute Scaled Error (Hyndman & Koehler, 2006).
    Normalise par l'erreur d'un modèle naïf saisonnier.
    MASE < 1 → meilleur que le naïf saisonnier.
    """
    naive_errors = np.abs(
        y_train[seasonality:].values - y_train[:-seasonality].values
    )
    scale = np.mean(naive_errors)
    if scale == 0:
        return np.nan
    return mae(y_true, y_pred) / scale


def peak_bias(y_true, y_pred, quantile=0.9):
    """
    Biais sur les pics de demande (> quantile).
    Négatif = on sous-estime les pics (problème opérationnel majeur).
    """
    # Guard : tableaux vides ou trop petits
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    mask = y_true >= np.quantile(y_true, quantile)
    if mask.sum() == 0:
        return np.nan
    return np.mean(y_pred[mask] - y_true[mask])


def evaluate_all(y_true, y_pred, y_train=None, label="Model"):
    """
    Calcule et affiche toutes les métriques d'un coup.
    
    Returns
    -------
    dict : dictionnaire des métriques
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE (%)": smape(y_true, y_pred),
        "MAPE (%)": mape(y_true, y_pred),
        "Peak Bias (top 10%)": peak_bias(y_true, y_pred, quantile=0.9),
    }
    if y_train is not None:
        metrics["MASE (24h)"] = mase(y_true, y_pred, pd.Series(y_train))

    print(f"\n{'='*45}")
    print(f"  Évaluation : {label}")
    print(f"{'='*45}")
    for k, v in metrics.items():
        print(f"  {k:<25} : {v:>8.3f}")
    print(f"{'='*45}\n")

    return metrics


# ─────────────────────────────────────────────────────────────
# HELPERS TEMPORELS
# ─────────────────────────────────────────────────────────────

def add_time_features(df, datetime_col="datetime"):
    """
    Ajoute des features temporelles à un DataFrame.
    Encode heure et jour de la semaine en sin/cos (cyclique).
    """
    dt = pd.to_datetime(df[datetime_col])

    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek  # 0=lundi
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["quarter"] = dt.dt.quarter

    # Encodage cyclique (capture la continuité : heure 23 proche de heure 0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def get_us_federal_holidays(years):
    """
    Retourne les jours fériés fédéraux américains pour les années données.
    Source : règles officielles (calcul sans dépendance externe).
    """
    holidays = []
    for year in years:
        # Jour de l'an
        holidays.append(pd.Timestamp(f"{year}-01-01"))
        # MLK Day : 3e lundi de janvier
        holidays.append(_nth_weekday(year, 1, 0, 3))
        # Presidents Day : 3e lundi de février
        holidays.append(_nth_weekday(year, 2, 0, 3))
        # Memorial Day : dernier lundi de mai
        holidays.append(_last_weekday(year, 5, 0))
        # Juneteenth
        holidays.append(pd.Timestamp(f"{year}-06-19"))
        # Independence Day
        holidays.append(pd.Timestamp(f"{year}-07-04"))
        # Labor Day : 1er lundi de septembre
        holidays.append(_nth_weekday(year, 9, 0, 1))
        # Columbus Day : 2e lundi d'octobre
        holidays.append(_nth_weekday(year, 10, 0, 2))
        # Veterans Day
        holidays.append(pd.Timestamp(f"{year}-11-11"))
        # Thanksgiving : 4e jeudi de novembre
        holidays.append(_nth_weekday(year, 11, 3, 4))
        # Christmas
        holidays.append(pd.Timestamp(f"{year}-12-25"))
    return set(holidays)


def _nth_weekday(year, month, weekday, n):
    """n-ième jour de la semaine (0=lundi) du mois."""
    first = pd.Timestamp(year=year, month=month, day=1)
    delta = (weekday - first.dayofweek) % 7
    first_occurrence = first + pd.Timedelta(days=delta)
    return first_occurrence + pd.Timedelta(weeks=n - 1)


def _last_weekday(year, month, weekday):
    """Dernier jour de la semaine (0=lundi) du mois."""
    last = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    delta = (last.dayofweek - weekday) % 7
    return last - pd.Timedelta(days=delta)


# ─────────────────────────────────────────────────────────────
# SPLIT TEMPOREL
# ─────────────────────────────────────────────────────────────

def temporal_split(df, datetime_col, train_end, val_end):
    """
    Split temporel strict — aucun data leakage possible.
    
    Parameters
    ----------
    df : DataFrame trié par datetime_col
    train_end : str, ex: '2024-09-30'
    val_end   : str, ex: '2024-10-31'
    
    Returns
    -------
    (train_df, val_df, test_df)
    """
    dt = pd.to_datetime(df[datetime_col])
    train = df[dt <= train_end].copy()
    val = df[(dt > train_end) & (dt <= val_end)].copy()
    test = df[dt > val_end].copy()

    print(f"Train : {len(train):>7,} lignes  ({dt[dt <= train_end].min().date()} → {train_end})")
    print(f"Val   : {len(val):>7,} lignes  ({train_end} → {val_end})")
    print(f"Test  : {len(test):>7,} lignes  ({val_end} → {dt[dt > val_end].max().date()})")

    return train, val, test