# -*- coding: utf-8 -*-

"""
02_eda.py — Analyse Exploratoire des Données (EDA)
===================================================
Génère des visualisations et statistiques descriptives.
Outputs sauvegardés dans data/processed/eda_figures/

Usage
-----
    python src/02_eda.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Style professionnel
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})
sns.set_palette("husl")

OUTPUT_DIR = "data/processed/eda_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────

def load_data():
    path = "data/processed/trips_hourly.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Données non trouvées. Lancer d'abord : python src/01_data_download.py"
        )
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["day_name"] = df["datetime"].dt.day_name()
    df["is_weekend"] = df["day_of_week"] >= 5
    return df


# ─────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────

def plot_serie_temporelle(df):
    """Série temporelle complète — tendance & saisonnalité annuelle."""
    daily = df.groupby("date")["total_rides"].sum().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["rolling_7d"] = daily["total_rides"].rolling(7, center=True).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily["date"], daily["total_rides"], alpha=0.3, color="steelblue", lw=0.8, label="Journalier")
    ax.plot(daily["date"], daily["rolling_7d"], color="steelblue", lw=2, label="Moyenne 7j")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.set_title("Demande quotidienne totale — Capital Bikeshare 2024", fontsize=14, fontweight="bold")
    ax.set_ylabel("Nombre de trajets")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_serie_temporelle.png")
    plt.close()
    print("  ✓ Série temporelle sauvegardée")


def plot_profil_horaire(df):
    """Profil moyen par heure × jour de semaine."""
    pivot = df.groupby(["hour", "is_weekend"])["total_rides"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    for is_we, label, color in [(False, "Semaine", "steelblue"), (True, "Week-end", "coral")]:
        sub = pivot[pivot["is_weekend"] == is_we]
        ax.plot(sub["hour"], sub["total_rides"], marker="o", label=label, color=color, lw=2)

    ax.set_xticks(range(24))
    ax.set_xlabel("Heure de la journée")
    ax.set_ylabel("Trajets moyens par heure")
    ax.set_title("Profil horaire moyen — Semaine vs Week-end", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_profil_horaire.png")
    plt.close()
    print("  ✓ Profil horaire sauvegardé")

    # Insight clé : double pic semaine (8h & 17h) = domination usage pendulaire
    peak_am = pivot[pivot["is_weekend"] == False].nlargest(1, "total_rides")
    print(f"     → Pic semaine : {peak_am['hour'].values[0]}h "
          f"({peak_am['total_rides'].values[0]:.0f} trajets/h en moy.)")


def plot_heatmap_dow_hour(df):
    """Heatmap : jour de semaine × heure — vue densité."""
    pivot = df.pivot_table(
        values="total_rides",
        index="day_of_week",
        columns="hour",
        aggfunc="mean"
    )
    day_labels = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd", linewidths=0.1,
        yticklabels=day_labels, cbar_kws={"label": "Trajets moyens/h"}
    )
    ax.set_title("Heatmap demande — Jour × Heure", fontsize=13, fontweight="bold")
    ax.set_xlabel("Heure")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_heatmap_dow_hour.png")
    plt.close()
    print("  ✓ Heatmap sauvegardée")


def plot_saisonnalite_mensuelle(df):
    """Box plots mensuels — saisonnalité et variabilité."""
    monthly_daily = df.groupby(["date", "month"])["total_rides"].sum().reset_index()
    month_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                    "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]

    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_daily.boxplot(column="total_rides", by="month", ax=ax,
                          patch_artist=True, showfliers=False)
    ax.set_xticklabels(month_labels[:df["month"].max()])
    ax.set_title("Distribution journalière par mois", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Trajets journaliers")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_saisonnalite_mensuelle.png")
    plt.close()
    print("  ✓ Saisonnalité mensuelle sauvegardée")


def plot_member_vs_casual(df):
    """Évolution part member vs casual — insight comportemental."""
    monthly = df.groupby("month")[["member_rides", "casual_rides"]].sum()
    monthly["pct_casual"] = 100 * monthly["casual_rides"] / (
        monthly["member_rides"] + monthly["casual_rides"]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    monthly[["member_rides", "casual_rides"]].plot(kind="bar", ax=ax1,
        color=["steelblue", "coral"], edgecolor="white")
    ax1.set_title("Trajets totaux : Members vs Casuals", fontweight="bold")
    ax1.set_ylabel("Nombre de trajets")
    ax1.set_xticklabels(
        ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"][:len(monthly)],
        rotation=45
    )

    ax2.plot(monthly.index, monthly["pct_casual"], marker="o", color="coral", lw=2)
    ax2.fill_between(monthly.index, monthly["pct_casual"], alpha=0.2, color="coral")
    ax2.set_title("% Casuals par mois (saisonnalité touristique)", fontweight="bold")
    ax2.set_ylabel("% Casuals")
    ax2.set_ylim(0, 50)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_member_vs_casual.png")
    plt.close()
    print("  ✓ Member vs Casual sauvegardé")


def print_stats_descriptives(df):
    """Résumé statistique complet."""
    print("\n" + "=" * 50)
    print("  STATISTIQUES DESCRIPTIVES")
    print("=" * 50)
    print(f"\n  Période couverte : {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  Nombre d'heures  : {len(df):,}")
    print(f"  Trajets totaux   : {df['total_rides'].sum():,}")
    print(f"\n  Demande horaire :")
    print(df["total_rides"].describe().to_string())
    print(f"\n  Heures avec 0 trajet : {(df['total_rides'] == 0).sum():,} "
          f"({100*(df['total_rides']==0).mean():.1f}%)")
    print(f"  Heure max observée : {df['total_rides'].max()} trajets")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n📊 Analyse Exploratoire des Données")
    print("=" * 50)

    df = load_data()
    print_stats_descriptives(df)

    print("\n🖼️  Génération des figures...")
    plot_serie_temporelle(df)
    plot_profil_horaire(df)
    plot_heatmap_dow_hour(df)
    plot_saisonnalite_mensuelle(df)
    plot_member_vs_casual(df)

    print(f"\n✅ Figures sauvegardées dans : {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()