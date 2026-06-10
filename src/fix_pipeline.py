# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 23:12:48 2026

@author: user
"""

# -*- coding: utf-8 -*-
"""
fix_pipeline.py — Diagnostic et correction complete du pipeline
===============================================================
Ce script :
  1. Diagnostique l'etat actuel des donnees
  2. Telecharge les mois manquants (juin-decembre 2024)
  3. Regenere features.parquet
  4. Relance baseline + XGBoost avec les bons splits
  5. Exporte tous les fichiers dashboard

Lancer depuis la racine du projet :
    python src/fix_pipeline.py

Prerequis :
    pip install requests requests-cache retry-requests xgboost joblib pyarrow
"""

import os, sys, warnings, zipfile, time, json
warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (mae, rmse, smape, evaluate_all,
                   temporal_split, add_time_features, get_us_federal_holidays)

# ─────────────────────────────────────────────────────────────
# CHEMINS
# ─────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
DASHBOARD_DIR = os.path.join(ROOT_DIR, "dashboard", "data")
MODELS_DIR    = os.path.join(ROOT_DIR, "models")
FIGURES_DIR   = os.path.join(PROCESSED_DIR, "eda_figures")

for d in [RAW_DIR, PROCESSED_DIR, DASHBOARD_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

BASE_URL = "https://s3.amazonaws.com/capitalbikeshare-data"

# Tous les mois necessaires (train=jan-sep, val=oct, test=nov-dec)
ALL_MONTHS = [
    ("2024", "01"), ("2024", "02"), ("2024", "03"),
    ("2024", "04"), ("2024", "05"), ("2024", "06"),
    ("2024", "07"), ("2024", "08"), ("2024", "09"),
    ("2024", "10"), ("2024", "11"), ("2024", "12"),
]

# ─────────────────────────────────────────────────────────────
# ETAPE 0 : DIAGNOSTIC
# ─────────────────────────────────────────────────────────────

def diagnostic():
    print("\n" + "="*55)
    print("  DIAGNOSTIC DE L'ETAT ACTUEL")
    print("="*55)

    hourly_path = os.path.join(PROCESSED_DIR, "trips_hourly.parquet")
    if os.path.exists(hourly_path):
        df = pd.read_parquet(hourly_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        print(f"\n  trips_hourly.parquet :")
        print(f"    Lignes  : {len(df):,}")
        print(f"    Periode : {df['datetime'].min().date()} -> {df['datetime'].max().date()}")
        print(f"    Mois couverts : {df['datetime'].dt.month.nunique()}")
        months_present = sorted(df["datetime"].dt.to_period("M").unique().astype(str))
        print(f"    Detail  : {months_present}")

        # Verifier si les donnees de test existent
        test_data = df[df["datetime"] >= "2024-11-01"]
        if len(test_data) == 0:
            print("\n  PROBLEME DETECTE : Pas de donnees apres octobre 2024 !")
            print("  -> Les scripts 04 et 05 produisent 0 predictions (jeu de test vide).")
            print("  -> Il faut telecharger les mois manquants.")
            return False
        else:
            print(f"\n  Donnees test (nov-dec) : {len(test_data):,} lignes OK")
            return True
    else:
        print("  trips_hourly.parquet introuvable")
        return False


# ─────────────────────────────────────────────────────────────
# ETAPE 1 : TELECHARGEMENT
# ─────────────────────────────────────────────────────────────

def download_month(year, month):
    filename = f"{year}{month}-capitalbikeshare-tripdata.zip"
    dest = os.path.join(RAW_DIR, filename)
    if os.path.exists(dest):
        print(f"  [CACHE] {filename}")
        return dest
    url = f"{BASE_URL}/{filename}"
    print(f"  [DL]    {filename} ...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=90)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        print(f"OK ({len(resp.content)/1e6:.1f} MB)")
        time.sleep(0.5)
        return dest
    except Exception as e:
        print(f"ERREUR : {e}")
        return None


def read_zip_to_df(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = [f for f in zf.namelist() if f.endswith(".csv")][0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)
    col_map = {
        "Start date":          "started_at",
        "End date":            "ended_at",
        "Member type":         "member_casual",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    keep = [c for c in ["started_at", "ended_at", "rideable_type",
                         "start_station_id", "end_station_id", "member_casual"]
            if c in df.columns]
    return df[keep]


def build_hourly(all_dfs):
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["started_at"] = pd.to_datetime(df_all["started_at"], errors="coerce")
    df_all = df_all.dropna(subset=["started_at"])
    df_all["datetime"] = df_all["started_at"].dt.floor("h")

    hourly = (
        df_all.groupby("datetime")
        .agg(
            total_rides  = ("started_at",    "count"),
            member_rides = ("member_casual",  lambda x: (x == "member").sum()),
            casual_rides = ("member_casual",  lambda x: (x == "casual").sum()),
        )
        .reset_index()
        .sort_values("datetime")
    )
    # Remplir heures manquantes
    full_range = pd.date_range(hourly["datetime"].min(),
                               hourly["datetime"].max(), freq="h")
    hourly = (hourly.set_index("datetime")
                    .reindex(full_range).fillna(0).reset_index()
                    .rename(columns={"index": "datetime"}))
    hourly[["total_rides","member_rides","casual_rides"]] = \
        hourly[["total_rides","member_rides","casual_rides"]].astype(int)
    return hourly


def step1_download_and_build():
    print("\n" + "="*55)
    print("  ETAPE 1 : TELECHARGEMENT DES DONNEES")
    print("="*55)

    all_dfs = []
    for year, month in ALL_MONTHS:
        zip_path = download_month(year, month)
        if zip_path is None:
            continue
        try:
            df = read_zip_to_df(zip_path)
            all_dfs.append(df)
            print(f"    -> {year}-{month} : {len(df):,} trajets bruts")
        except Exception as e:
            print(f"  WARN lecture {year}-{month} : {e}")

    if not all_dfs:
        raise RuntimeError("Aucun fichier disponible.")

    print(f"\n  Agregation horaire de {len(all_dfs)} mois...")
    hourly = build_hourly(all_dfs)
    out = os.path.join(PROCESSED_DIR, "trips_hourly.parquet")
    hourly.to_parquet(out, index=False)
    print(f"  Sauvegarde : {out}")
    print(f"  Shape : {hourly.shape}")
    print(f"  Periode : {hourly['datetime'].min().date()} -> {hourly['datetime'].max().date()}")
    return hourly


# ─────────────────────────────────────────────────────────────
# ETAPE 2 : FEATURE ENGINEERING (sans API meteo pour simplicite)
# ─────────────────────────────────────────────────────────────

def add_lag_features(df, target_col="total_rides", horizon=48):
    df = df.sort_values("datetime").reset_index(drop=True)
    for lag_h in [horizon, 7*24, 14*24, 21*24]:
        df[f"lag_{lag_h}h"] = df[target_col].shift(lag_h)
    shifted = df[target_col].shift(horizon)
    for window in [24, 48, 7*24]:
        df[f"rolling_mean_{window}h"] = shifted.rolling(window, min_periods=window//2).mean()
        df[f"rolling_std_{window}h"]  = shifted.rolling(window, min_periods=window//2).std()
    df["lag_24h"] = df[target_col].shift(24)
    return df


def add_holiday_features(df):
    years = df["datetime"].dt.year.unique().tolist()
    holidays = get_us_federal_holidays(years)
    df["date_only"] = df["datetime"].dt.normalize()
    df["is_holiday"] = df["date_only"].isin(holidays).astype(int)
    hd = pd.DatetimeIndex(list(holidays))
    df["is_holiday_eve"]   = df["date_only"].isin(set((hd - pd.Timedelta(days=1)).normalize())).astype(int)
    df["is_holiday_after"] = df["date_only"].isin(set((hd + pd.Timedelta(days=1)).normalize())).astype(int)
    df = df.drop(columns=["date_only"])
    return df


def step2_feature_engineering(hourly):
    print("\n" + "="*55)
    print("  ETAPE 2 : FEATURE ENGINEERING")
    print("="*55)

    df = hourly.copy()
    df = add_time_features(df, datetime_col="datetime")
    df = add_lag_features(df, target_col="total_rides", horizon=48)
    df = add_holiday_features(df)

    # Supprimer les NaN sur les lags essentiels
    df_clean = df.dropna(subset=["lag_48h", "lag_168h"]).copy()
    out = os.path.join(PROCESSED_DIR, "features.parquet")
    df_clean.to_parquet(out, index=False)
    print(f"  Shape finale : {df_clean.shape}")
    print(f"  Periode : {df_clean['datetime'].min().date()} -> {df_clean['datetime'].max().date()}")
    print(f"  Sauvegarde : {out}")
    return df_clean


# ─────────────────────────────────────────────────────────────
# ETAPE 3 : BASELINE
# ─────────────────────────────────────────────────────────────

def step3_baseline(df):
    print("\n" + "="*55)
    print("  ETAPE 3 : MODELE BASELINE")
    print("="*55)

    df = df.sort_values("datetime").reset_index(drop=True)

    print("\n  Split temporel :")
    train, val, test = temporal_split(df, "datetime",
                                      train_end="2024-09-30",
                                      val_end="2024-10-31")
    train = train.reset_index(drop=True)
    val   = val.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    print(f"\n  Verification splits :")
    print(f"    Train : {len(train):,} | Val : {len(val):,} | Test : {len(test):,}")

    if len(test) == 0:
        print("\n  ERREUR : Le jeu de test est vide !")
        print("  Verifiez que les donnees couvrent bien nov-dec 2024.")
        return None, None

    full_series = df["total_rides"]
    val_dates   = set(val["datetime"])
    test_dates  = set(test["datetime"])

    # Naif saisonnier
    pred_naive_all  = full_series.shift(168)
    pred_naive_val  = pred_naive_all[df["datetime"].isin(val_dates)].values
    pred_naive_test = pred_naive_all[df["datetime"].isin(test_dates)].values

    # Moyenne ponderee
    weights = np.array([1, 2, 3, 4], dtype=float)
    weights /= weights.sum()
    pred_wwa_all = pd.Series(np.nan, index=df.index)
    for i in range(4*168, len(full_series)):
        lags = [full_series.iloc[i - k*168] for k in range(1, 5)]
        pred_wwa_all.iloc[i] = np.dot(lags, weights[::-1])
    pred_wwa_val  = pred_wwa_all[df["datetime"].isin(val_dates)].values
    pred_wwa_test = pred_wwa_all[df["datetime"].isin(test_dates)].values

    m_naive_test = evaluate_all(test["total_rides"].values, pred_naive_test,
                                y_train=train["total_rides"], label="Naif — Test")
    m_wwa_test   = evaluate_all(test["total_rides"].values, pred_wwa_test,
                                y_train=train["total_rides"], label="Moy. ponderee — Test")

    # Sauvegarde
    preds_df = test[["datetime", "total_rides"]].copy()
    preds_df["pred_naive"] = pred_naive_test
    preds_df["pred_wwa"]   = pred_wwa_test
    out = os.path.join(PROCESSED_DIR, "baseline_predictions.parquet")
    preds_df.to_parquet(out, index=False)
    print(f"  Sauvegarde : {out}  ({len(preds_df):,} lignes)")

    return preds_df, {"naive": m_naive_test, "wwa": m_wwa_test}


# ─────────────────────────────────────────────────────────────
# ETAPE 4 : XGBOOST
# ─────────────────────────────────────────────────────────────

TEMPORAL_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "is_weekend",
    "is_holiday", "is_holiday_eve", "is_holiday_after", "day_of_year",
]
LAG_FEATURES = [
    "lag_48h", "lag_168h", "lag_336h", "lag_504h",
    "rolling_mean_24h", "rolling_mean_48h", "rolling_mean_168h",
    "rolling_std_24h", "rolling_std_48h",
]
TARGET = "total_rides"


def step4_xgboost(df):
    print("\n" + "="*55)
    print("  ETAPE 4 : XGBOOST")
    print("="*55)

    try:
        import xgboost as xgb
    except ImportError:
        print("  xgboost non installe. Lancer : pip install xgboost")
        return None, None

    df = df.sort_values("datetime").reset_index(drop=True)
    train, val, test = temporal_split(df, "datetime",
                                      train_end="2024-09-30",
                                      val_end="2024-10-31")

    features = [f for f in TEMPORAL_FEATURES + LAG_FEATURES if f in df.columns]
    print(f"\n  Features : {len(features)}")

    X_train = train[features].fillna(0)
    X_val   = val[features].fillna(0)
    X_test  = test[features].fillna(0)
    y_train = train[TARGET]
    y_val   = val[TARGET]
    y_test  = test[TARGET]

    print("  Entrainement XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05,
        max_depth=6, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
        eval_metric="mae", early_stopping_rounds=50,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Arbres : {model.best_iteration}")

    pred_test = np.maximum(model.predict(X_test), 0)
    m_test = evaluate_all(y_test.values, pred_test,
                          y_train=y_train, label="XGBoost — Test")

    preds_df = test[["datetime", TARGET]].copy()
    preds_df["pred_xgb"] = pred_test
    preds_df["residual"] = preds_df[TARGET] - pred_test
    out = os.path.join(PROCESSED_DIR, "xgb_predictions.parquet")
    preds_df.to_parquet(out, index=False)
    print(f"  Sauvegarde : {out}  ({len(preds_df):,} lignes)")

    joblib.dump({"model": model, "feature_cols": features, "metrics_test": m_test},
                os.path.join(MODELS_DIR, "xgb_model.joblib"))

    return preds_df, m_test


# ─────────────────────────────────────────────────────────────
# ETAPE 5 : EXPORT DASHBOARD
# ─────────────────────────────────────────────────────────────

def step5_dashboard(hourly, features_df, baseline_preds, xgb_preds, m_xgb, m_naive):
    print("\n" + "="*55)
    print("  ETAPE 5 : EXPORT DASHBOARD")
    print("="*55)

    # ── Table principale
    main = xgb_preds.merge(
        baseline_preds[["datetime", "pred_naive", "pred_wwa"]],
        on="datetime", how="left"
    )
    feat_cols = ["datetime", "hour", "day_of_week", "month", "is_weekend", "is_holiday"]
    feat_cols = [c for c in feat_cols if c in features_df.columns]
    main = main.merge(features_df[feat_cols], on="datetime", how="left")

    main["date"]  = main["datetime"].dt.date
    main["hour"]  = main["datetime"].dt.hour
    main["month"] = main["datetime"].dt.month
    main["dow"]   = main["datetime"].dt.dayofweek
    main["week"]  = main["datetime"].dt.isocalendar().week.astype(int)

    main["err_naive"] = (main["total_rides"] - main["pred_naive"].fillna(0)).abs()
    main["err_wwa"]   = (main["total_rides"] - main["pred_wwa"].fillna(0)).abs()
    main["err_xgb"]   = (main["total_rides"] - main["pred_xgb"]).abs()

    denom = main["total_rides"].abs() + main["pred_xgb"].abs() + 1e-8
    main["smape_xgb"] = 200 * main["err_xgb"] / denom

    main["is_peak"] = (main["total_rides"] >= main["total_rides"].quantile(0.90)).astype(int)
    main = main.sort_values("datetime").reset_index(drop=True)
    main["mae_rolling_24h"] = main["err_xgb"].rolling(24, min_periods=1).mean()
    print(f"  main : {main.shape}")

    # ── Agregation journaliere
    daily_tmp = hourly.copy()
    daily_tmp["date"]  = daily_tmp["datetime"].dt.date
    daily_tmp["month"] = daily_tmp["datetime"].dt.month
    daily_tmp["dow"]   = daily_tmp["datetime"].dt.dayofweek
    daily_tmp["week"]  = daily_tmp["datetime"].dt.isocalendar().week.astype(int)

    daily = daily_tmp.groupby("date").agg(
        total_rides  = ("total_rides",  "sum"),
        member_rides = ("member_rides", "sum"),
        casual_rides = ("casual_rides", "sum"),
        month=("month", "first"), dow=("dow", "first"), week=("week", "first"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    years = daily["date"].dt.year.unique().tolist()
    holidays = get_us_federal_holidays(years)
    daily["is_holiday"] = daily["date"].isin(holidays).astype(int)

    test_daily = main.groupby("date").agg(
        pred_naive=("pred_naive","sum"), pred_wwa=("pred_wwa","sum"), pred_xgb=("pred_xgb","sum")
    ).reset_index()
    test_daily["date"] = pd.to_datetime(test_daily["date"])
    daily = daily.merge(test_daily, on="date", how="left")
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["pct_change"] = daily["total_rides"].pct_change() * 100
    print(f"  daily : {daily.shape}")

    # ── Profil horaire
    hw = hourly.copy()
    hw["hour"]       = hw["datetime"].dt.hour
    hw["dow"]        = hw["datetime"].dt.dayofweek
    hw["month"]      = hw["datetime"].dt.month
    hw["is_weekend"] = (hw["dow"] >= 5).astype(int)
    hw["date_n"]     = hw["datetime"].dt.normalize()
    hw["is_holiday"] = hw["date_n"].isin(holidays).astype(int)

    def classify(row):
        if row["is_holiday"]:   return "Jour ferie"
        elif row["is_weekend"]: return "Week-end"
        else:                   return "Semaine"

    hw["day_type"] = hw.apply(classify, axis=1)

    profile = hw.groupby(["hour","day_type"])["total_rides"].agg(
        ["mean","std","median","count"]
    ).reset_index()
    profile.columns = ["hour","day_type","mean_rides","std_rides","median_rides","n"]
    profile["ci_low"]  = profile["mean_rides"] - 1.96*profile["std_rides"]/np.sqrt(profile["n"])
    profile["ci_high"] = profile["mean_rides"] + 1.96*profile["std_rides"]/np.sqrt(profile["n"])

    profile_monthly = hw.groupby(["hour","month"])["total_rides"].mean().reset_index()
    profile_monthly.columns = ["hour","month","mean_rides"]
    print(f"  profile : {profile.shape}")

    # ── Robustesse
    def seg_metrics(mask, label):
        sub = main[mask].copy()
        valid = sub[["total_rides","pred_xgb","pred_naive"]].dropna()
        if len(valid) < 5: return None
        y = valid["total_rides"].values
        p = valid["pred_xgb"].values
        n = valid["pred_naive"].fillna(0).values
        mae_xgb  = float(np.mean(np.abs(y - p)))
        mae_naif = float(np.mean(np.abs(y - n)))
        return {
            "segment":     label,
            "n":           len(valid),
            "mean_demand": round(float(y.mean()), 1),
            "MAE_naive":   round(mae_naif, 1),
            "MAE_xgb":     round(mae_xgb,  1),
            "sMAPE_xgb":   round(float(np.mean(200*np.abs(y-p)/(np.abs(y)+np.abs(p)+1e-8))), 1),
            "bias_xgb":    round(float((p - y).mean()), 1),
        }

    segs = []
    hol  = main.get("is_holiday", pd.Series(0, index=main.index))
    for label, mask in [
        ("Semaine normale", (main["dow"] < 5) & (hol == 0)),
        ("Week-end",         main["dow"] >= 5),
        ("Jour ferie",       hol == 1),
        ("Rush matin (7-9h)", main["hour"].isin(range(7,10))),
        ("Journee (10-16h)",  main["hour"].isin(range(10,17))),
        ("Rush soir (17-19h)",main["hour"].isin(range(17,20))),
        ("Nuit (0-6h)",       main["hour"].isin(range(0,7))),
        ("Pic (top 10%)",     main["is_peak"] == 1),
        ("Normal (mid 50%)",  (main["total_rides"] >= main["total_rides"].quantile(0.25)) &
                              (main["total_rides"] <  main["total_rides"].quantile(0.75))),
        ("Creux (bot 10%)",   main["total_rides"] <= main["total_rides"].quantile(0.10)),
    ]:
        r = seg_metrics(mask, label)
        if r: segs.append(r)

    robust = pd.DataFrame(segs)
    if not robust.empty:
        robust["gain_pct"] = (
            100*(robust["MAE_naive"] - robust["MAE_xgb"]) / robust["MAE_naive"]
        ).round(1)
    print(f"  robustesse : {robust.shape}")

    # ── KPIs
    y_true = main["total_rides"].values
    p_xgb  = main["pred_xgb"].values
    p_naif = main["pred_naive"].fillna(0).values

    mae_x = float(np.mean(np.abs(y_true - p_xgb)))
    mae_n = float(np.mean(np.abs(y_true - p_naif)))

    def safe_round(v, n=2):
        try:
            f = float(v)
            if np.isnan(f): return None
            return round(f, n)
        except: return None

    kpis = {
        "total_rides_all":      int(hourly["total_rides"].sum()),
        "total_rides_test":     int(main["total_rides"].sum()),
        "n_hours_total":        int(len(hourly)),
        "n_hours_test":         int(len(main)),
        "period_start":         str(hourly["datetime"].min().date()),
        "period_end":           str(hourly["datetime"].max().date()),
        "test_start":           str(main["datetime"].min().date()),
        "test_end":             str(main["datetime"].max().date()),
        "xgb_mae":              safe_round(mae_x),
        "xgb_rmse":             safe_round(np.sqrt(np.mean((y_true - p_xgb)**2))),
        "xgb_smape":            safe_round(np.mean(200*np.abs(y_true-p_xgb)/(np.abs(y_true)+np.abs(p_xgb)+1e-8))),
        "naive_mae":            safe_round(mae_n),
        "naive_smape":          safe_round(np.mean(200*np.abs(y_true-p_naif)/(np.abs(y_true)+np.abs(p_naif)+1e-8))),
        "improvement_mae_pct":  safe_round(100*(mae_n - mae_x)/mae_n, 1) if mae_n > 0 else None,
        "avg_hourly_demand":    safe_round(float(hourly["total_rides"].mean()), 1),
        "peak_demand":          int(hourly["total_rides"].max()),
        "peak_hour_threshold":  safe_round(float(main["total_rides"].quantile(0.90)), 1),
        "pct_member":           safe_round(100*hourly["member_rides"].sum()/hourly["total_rides"].sum(), 1),
        "pct_casual":           safe_round(100*hourly["casual_rides"].sum()/hourly["total_rides"].sum(), 1),
    }
    print(f"\n  KPIs :")
    for k, v in kpis.items():
        print(f"    {k:<35} : {v}")

    # ── Export
    print("\n  Export des fichiers :")
    exports = [
        (main,            "dashboard_main.parquet"),
        (daily,           "dashboard_daily.parquet"),
        (profile,         "dashboard_hourly_profile.parquet"),
        (profile_monthly, "dashboard_hourly_monthly.parquet"),
        (robust,          "dashboard_robustness.parquet"),
    ]
    for df_exp, fname in exports:
        path = os.path.join(DASHBOARD_DIR, fname)
        df_exp.to_parquet(path, index=False)
        kb = os.path.getsize(path) // 1024
        print(f"    OK {fname:<45} ({len(df_exp):>5,} lignes, {kb} KB)")

    kpi_path = os.path.join(DASHBOARD_DIR, "dashboard_metrics.json")
    with open(kpi_path, "w", encoding="utf-8") as f:
        json.dump(kpis, f, indent=2, ensure_ascii=True)
    print(f"    OK dashboard_metrics.json")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  FIX PIPELINE — Capital Bikeshare")
    print("="*55)

    data_ok = diagnostic()

    if not data_ok:
        print("\n  -> Telechargement des donnees manquantes...")
        hourly = step1_download_and_build()
    else:
        print("\n  -> Donnees OK, chargement depuis le cache...")
        hourly = pd.read_parquet(os.path.join(PROCESSED_DIR, "trips_hourly.parquet"))
        hourly["datetime"] = pd.to_datetime(hourly["datetime"])

    features_df = step2_feature_engineering(hourly)
    baseline_preds, m_baseline = step3_baseline(features_df)

    if baseline_preds is None:
        print("\n  ECHEC : impossible de continuer sans donnees de test.")
        return

    xgb_preds, m_xgb = step4_xgboost(features_df)

    if xgb_preds is None:
        print("\n  ECHEC : XGBoost n'a pas produit de predictions.")
        return

    step5_dashboard(hourly, features_df, baseline_preds, xgb_preds, m_xgb, m_baseline)

    print("\n" + "="*55)
    print("  PIPELINE TERMINE AVEC SUCCES")
    print("="*55)
    print("\n  Lancer le dashboard :")
    print("    python dashboard/app.py")
    print("    -> http://127.0.0.1:8050\n")


if __name__ == "__main__":
    main()