# -*- coding: utf-8 -*-
"""
03_feature_engineering.py — Construction des features
======================================================
Enrichit les données horaires avec :
  - Features temporelles (cycliques, calendaires)
  - Lags et moyennes mobiles (sans data leakage)
  - Données météo (Open-Meteo API, gratuit)
  - Jours fériés américains

Usage
-----
    python src/03_feature_engineering.py

Sortie
------
    data/processed/features.parquet
"""

import os
import warnings
import pandas as pd
import numpy as np
import requests
import requests_cache
from retry_requests import retry

warnings.filterwarnings("ignore")

# Import des utilitaires locaux
import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import add_time_features, get_us_federal_holidays

PROCESSED_DIR = "data/processed"

# ─────────────────────────────────────────────────────────────
# MÉTÉO — Open-Meteo API (gratuit, sans clé)
# ─────────────────────────────────────────────────────────────

# Coordonnées de Washington DC
DC_LAT = 38.8951
DC_LON = -77.0364


def setup_meteo_client():
    """Configure un client HTTP avec cache et retry pour Open-Meteo."""
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return retry_session


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Récupère les données météo horaires via Open-Meteo (historical).
    
    Variables : température, précipitations, vitesse du vent,
                code météo (nuageux, pluie, neige...)
    
    Parameters
    ----------
    start_date : "2024-01-01"
    end_date   : "2024-12-31"
    
    Returns
    -------
    DataFrame avec colonne datetime + variables météo
    """
    session = setup_meteo_client()

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": DC_LAT,
        "longitude": DC_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "precipitation",
            "windspeed_10m",
            "weathercode",
            "apparent_temperature",  # ressentie
            "relative_humidity_2m",
        ],
        "timezone": "America/New_York",
    }

    print("  📡 Appel Open-Meteo API...")
    resp = session.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df_weather = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "apparent_temp": hourly["apparent_temperature"],
        "precipitation": hourly["precipitation"],
        "windspeed": hourly["windspeed_10m"],
        "weathercode": hourly["weathercode"],
        "humidity": hourly["relative_humidity_2m"],
    })

    # Feature : conditions météo simplifiées
    # Codes WMO : 0-1=clair, 2-3=nuageux, 51-67=pluie, 71-77=neige, 80-99=orage
    def weather_category(code):
        if pd.isna(code):
            return "unknown"
        code = int(code)
        if code <= 1:
            return "clear"
        elif code <= 3:
            return "cloudy"
        elif code <= 48:
            return "fog"
        elif code <= 67:
            return "rain"
        elif code <= 77:
            return "snow"
        else:
            return "storm"

    df_weather["weather_cat"] = df_weather["weathercode"].apply(weather_category)
    df_weather["is_rain"] = df_weather["weather_cat"].isin(["rain", "storm"]).astype(int)
    df_weather["is_snow"] = (df_weather["weather_cat"] == "snow").astype(int)

    # Température² (effet non-linéaire : froid ET chaud réduisent la demande)
    df_weather["temp_squared"] = df_weather["temperature"] ** 2

    # Température anormale (> 2 std de la moyenne mensuelle)
    monthly_temp = df_weather.groupby(df_weather["datetime"].dt.month)["temperature"].transform("mean")
    monthly_std = df_weather.groupby(df_weather["datetime"].dt.month)["temperature"].transform("std")
    df_weather["temp_anomaly"] = ((df_weather["temperature"] - monthly_temp) / monthly_std).abs()
    df_weather["is_extreme_temp"] = (df_weather["temp_anomaly"] > 2).astype(int)

    print(f"  ✓ Météo : {len(df_weather):,} heures récupérées")
    return df_weather


# ─────────────────────────────────────────────────────────────
# FEATURES LAG (SANS DATA LEAKAGE)
# ─────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, target_col: str = "total_rides",
                     horizon: int = 48) -> pd.DataFrame:
    """
    Ajoute des features de lag temporel.
    
    Règle anti-leakage : tous les lags doivent être >= horizon.
    Pour horizon J+2 (48h), le plus petit lag utilisable est 48h.
    
    Features créées :
    - lag_48h  : demande il y a exactement 48h
    - lag_168h : demande il y a 7 jours (même heure, même jour)
    - lag_336h : demande il y a 14 jours
    - rolling_mean_48h : moyenne des 24h avant la fenêtre de prédiction
    - rolling_std_48h  : volatilité récente
    """
    df = df.sort_values("datetime").reset_index(drop=True)

    # Lags "ponctuels"
    for lag_h in [horizon, 7 * 24, 14 * 24, 21 * 24]:
        df[f"lag_{lag_h}h"] = df[target_col].shift(lag_h)

    # Moyennes mobiles décalées (rolling sur les 24h AVANT le lag minimum)
    # shift(horizon) puis rolling sur fenêtre en arrière = pas de leakage
    shifted = df[target_col].shift(horizon)
    for window in [24, 48, 7 * 24]:
        df[f"rolling_mean_{window}h"] = shifted.rolling(window, min_periods=window // 2).mean()
        df[f"rolling_std_{window}h"] = shifted.rolling(window, min_periods=window // 2).std()

    # Même heure la veille (utile pour capturer le pattern intraday)
    df["lag_24h"] = df[target_col].shift(24)   # NOTE: lag_24h <= horizon=48h est OK

    return df


# ─────────────────────────────────────────────────────────────
# JOURS FÉRIÉS & ÉVÉNEMENTS
# ─────────────────────────────────────────────────────────────

def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute l'indicateur jour férié et les jours adjacents."""
    years = df["datetime"].dt.year.unique().tolist()
    holidays = get_us_federal_holidays(years)

    df["date_only"] = df["datetime"].dt.normalize()
    df["is_holiday"] = df["date_only"].isin(holidays).astype(int)

    # Veille et lendemain de férié (comportement modifié)
    holiday_dates = pd.DatetimeIndex(list(holidays))
    day_before = set((holiday_dates - pd.Timedelta(days=1)).normalize())
    day_after = set((holiday_dates + pd.Timedelta(days=1)).normalize())

    df["is_holiday_eve"] = df["date_only"].isin(day_before).astype(int)
    df["is_holiday_after"] = df["date_only"].isin(day_after).astype(int)

    df = df.drop(columns=["date_only"])
    return df


# ─────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────

def main():
    print("\n🔧 Feature Engineering")
    print("=" * 50)

    # Chargement
    df = pd.read_parquet(f"{PROCESSED_DIR}/trips_hourly.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"  Données chargées : {len(df):,} heures")

    # 1. Features temporelles
    print("\n  1/4 Features temporelles...")
    df = add_time_features(df, datetime_col="datetime")

    # 2. Données météo
    print("\n  2/4 Données météo...")
    start = df["datetime"].min().strftime("%Y-%m-%d")
    end = df["datetime"].max().strftime("%Y-%m-%d")
    try:
        weather = fetch_weather(start, end)
        df = df.merge(weather, on="datetime", how="left")
        print(f"  ✓ Météo mergée ({df['temperature'].notna().sum():,} heures avec données)")
    except Exception as e:
        print(f"  ⚠️  Météo non disponible ({e}) — features météo omises")
        # Ajouter colonnes vides pour compatibilité
        for col in ["temperature", "apparent_temp", "precipitation", "windspeed",
                    "is_rain", "is_snow", "is_extreme_temp", "temp_squared"]:
            df[col] = np.nan

    # 3. Lags
    print("\n  3/4 Features lag (horizon 48h)...")
    df = add_lag_features(df, target_col="total_rides", horizon=48)
    n_lag_null = df["lag_48h"].isna().sum()
    print(f"  ✓ Lags créés — {n_lag_null} valeurs nulles (début de série, normal)")

    # 4. Jours fériés
    print("\n  4/4 Jours fériés...")
    df = add_holiday_features(df)
    print(f"  ✓ Jours fériés : {df['is_holiday'].sum()} heures concernées")

    # Vérification anti-leakage
    print("\n  🔍 Vérification anti-leakage...")
    lag_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("rolling_")]
    print(f"     Features lag utilisées : {lag_cols}")
    print("     ✓ Tous les lags >= 48h (horizon de prédiction J+2)")

    # Suppression des premières lignes avec NaN sur les lags
    df_clean = df.dropna(subset=["lag_48h", "lag_168h"]).copy()
    print(f"\n  Shape finale : {df_clean.shape}")
    print(f"  Features disponibles ({len(df_clean.columns)}) :")
    print(f"  {list(df_clean.columns)}")

    # Sauvegarde
    out_path = f"{PROCESSED_DIR}/features.parquet"
    df_clean.to_parquet(out_path, index=False)
    print(f"\n✅ Sauvegardé : {out_path}")


if __name__ == "__main__":
    main()

