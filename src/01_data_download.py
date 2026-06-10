# -*- coding: utf-8 -*-

"""
01_data_download.py — Téléchargement & préparation des données
==============================================================
Télécharge les données mensuelles Capital Bikeshare pour 2024 et 2025,
les concatène, et produit un fichier agrégé par heure.

Usage
-----
    python src/01_data_download.py

Sortie
------
    data/raw/       : fichiers ZIP bruts (un par mois)
    data/processed/trips_hourly.parquet : agrégat horaire prêt à modéliser
"""
# 
import os
import zipfile
import io
import time

import requests
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

BASE_URL = "https://s3.amazonaws.com/capitalbikeshare-data"

# Mois à télécharger : Jan 2024 → Dec 2024 + Jan-Mar 2025
MONTHS = []

for year in [2024, 2025]:
    for month in range(1, 13):
        MONTHS.append((str(year), f"{month:02d}"))

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


# ─────────────────────────────────────────────────────────────
# FONCTIONS
# ─────────────────────────────────────────────────────────────

def download_month(year: str, month: str, raw_dir: str) -> str | None:
    """
    Télécharge le ZIP d'un mois si pas déjà présent.
    Retourne le chemin du ZIP, ou None si erreur.
    """
    filename = f"{year}{month}-capitalbikeshare-tripdata.zip"
    url = f"{BASE_URL}/{filename}"
    dest = os.path.join(raw_dir, filename)

    if os.path.exists(dest):
        print(f"  [CACHE] {filename}")
        return dest

    print(f"  [DL]    {filename} ...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        print(f"OK ({len(resp.content) / 1e6:.1f} MB)")
        time.sleep(0.5)  # politesse envers le serveur
        return dest
    except requests.HTTPError as e:
        print(f"ERREUR : {e}")
        return None


def read_zip_to_df(zip_path: str) -> pd.DataFrame:
    """
    Lit le CSV dans un ZIP et retourne un DataFrame.
    Gère plusieurs variantes de noms de colonnes Capital Bikeshare.
    """
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = [f for f in zf.namelist() if f.endswith(".csv")][0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)

    # Normalisation des noms de colonnes (Capital Bikeshare a changé de schéma)
    col_map = {
        # Ancien schéma (pré-2020)
        "Start date": "started_at",
        "End date": "ended_at",
        "Start station number": "start_station_id",
        "End station number": "end_station_id",
        "Member type": "member_casual",
        # Nouveau schéma (2020+) — déjà au bon format
        "started_at": "started_at",
        "ended_at": "ended_at",
        "member_casual": "member_casual",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Garder uniquement les colonnes utiles
    keep = [c for c in ["started_at", "ended_at", "rideable_type",
                         "start_station_id", "end_station_id",
                         "member_casual"] if c in df.columns]
    return df[keep]


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df["started_at"] = pd.to_datetime(
        df["started_at"],
        format="mixed",
        errors="coerce"
        )
    
    print("Lignes après conversion datetime :", len(df))

    print(
        "NaT:",
        df["started_at"].isna().sum()
        )
    df = df.dropna(subset=["started_at"])

    df["datetime"] = df["started_at"].dt.floor("h")
    
    print(df["started_at"].dtype)
    print(df["started_at"].head())
    print(df["started_at"].tail())
    
    print(
        df["datetime"].min(),
        df["datetime"].max()
        )

    print(
        df["datetime"].dt.to_period("M")
        .value_counts()
        .sort_index()
        )

    hourly = (
        df.groupby("datetime")
        .agg(
            total_rides=("started_at", "count"),
            member_rides=("member_casual", lambda x: (x == "member").sum()),
            casual_rides=("member_casual", lambda x: (x == "casual").sum()),
        )
        .reset_index()
    )

    # 🔥 grille FIXE
    full_range = pd.date_range(
        start="2024-01-01 00:00:00",
        end="2025-12-31 23:00:00",
        freq="h"
    )

    hourly = (
        hourly.set_index("datetime")
        .reindex(full_range, fill_value=0)
        .reset_index()
        .rename(columns={"index": "datetime"})
    )

    return hourly


# ─────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("\n📥 Téléchargement des données Capital Bikeshare")
    print("=" * 50)

    all_dfs = []
    for year, month in tqdm(MONTHS, desc="Mois"):
        zip_path = download_month(year, month, RAW_DIR)
        if zip_path is None:
            continue
        try:
            df = read_zip_to_df(zip_path)
            print(
                f"{year}-{month} : "
                f"{len(df):,} trajets"
                )
            all_dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Erreur lecture {year}-{month} : {e}")

    if not all_dfs:
        raise RuntimeError("Aucun fichier téléchargé avec succès.")

    print(f"\n🔀 Concaténation de {len(all_dfs)} mois...")
    df_all = pd.concat(all_dfs, ignore_index=True)
    print(df_all["started_at"].head())
    print(df_all["started_at"].tail())
    print(f"   Total brut : {len(df_all):,} trajets")

    print("\n⏱️  Agrégation horaire...")
    hourly = aggregate_hourly(df_all)
    print(f"   Total horaire : {len(hourly):,} lignes")
    print(f"   Période : {hourly['datetime'].min()} → {hourly['datetime'].max()}")
    print(f"   Stats demande :")
    print(hourly["total_rides"].describe().to_string())

    # Remplir les heures manquantes avec 0 (nuits creuses, etc.)
    full_range = pd.date_range(
        start=hourly["datetime"].min(),
        end=hourly["datetime"].max(),
        freq="h"
    )
    hourly = (
        hourly.set_index("datetime")
        .reindex(full_range)
        .fillna(0)
        .reset_index()
        .rename(columns={"index": "datetime"})
    )
    """"""
    hourly[["total_rides", "member_rides", "casual_rides"]] = (
        hourly[["total_rides", "member_rides", "casual_rides"]].astype(int)
    )

    # Sauvegarde
    output_path = os.path.join(PROCESSED_DIR, "trips_hourly.parquet")
    hourly.to_parquet(output_path, index=False)
    print(f"\n✅ Sauvegardé : {output_path}")
    print(f"   Shape finale : {hourly.shape}")


if __name__ == "__main__":
    main()