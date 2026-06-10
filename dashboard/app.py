# -*- coding: utf-8 -*-
"""
dashboard/app.py — Dashboard Capital Bikeshare Demand Forecasting
=================================================================
Dashboard interactif Plotly Dash avec 4 onglets :

  1. Vue Generale   : serie temporelle, KPIs, profil horaire, member vs casual
  2. Predictions    : comparaison reel vs predit, slider fenetre, heatmap, erreurs
  3. Robustesse     : MAE par segment, gain vs baseline, tableau detaille
  4. Analyse EDA    : heatmap heure x jour, heatmap mois x heure, stats cles

Corrections apportees :
  - hex_to_rgba() remplace les couleurs #RRGGBBAA incompatibles avec Plotly
  - flask_caching pour performance
  - dcc.Loading spinners
  - Toutes les figures completement implementees
  - Compatible Windows / Spyder

Usage
-----
    python dashboard/app.py
    Ouvrir http://127.0.0.1:8050

Prerequis
---------
    pip install dash dash-bootstrap-components flask-caching plotly pandas pyarrow
    Notebook 03_Dashboard_Prep.ipynb doit avoir ete execute.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from flask_caching import Cache

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

COLORS = {
    "primary":   "#2C6EF2",
    "secondary": "#FF6B35",
    "success":   "#2DC653",
    "warning":   "#FFC107",
    "danger":    "#E63946",
    "dark":      "#1A1A2E",
    "card_bg":   "#16213E",
    "text":      "#E8E8E8",
    "subtext":   "#9E9E9E",
    "grid":      "#2A2A4A",
    "real":      "#64DFDF",
    "naive":     "#F4A261",
    "xgb":       "#E63946",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
        yaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", borderwidth=0),
        margin=dict(l=50, r=20, t=40, b=40),
    )
)


def hex_to_rgba(hex_color, alpha=0.15):
    """Convertit #RRGGBB -> rgba(r,g,b,alpha) compatible Plotly."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────────────────────────────────────────────
# APPLICATION + CACHE
# ─────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Bikeshare Forecast Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server

cache = Cache(
    server,
    config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300}
)


# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNEES (mis en cache)
# ─────────────────────────────────────────────────────────────

@cache.memoize()
def load_data():
    """Charge tous les fichiers prepares par le notebook 03."""
    files = {
        "main":      "dashboard_main.parquet",
        "daily":     "dashboard_daily.parquet",
        "profile":   "dashboard_hourly_profile.parquet",
        "profile_m": "dashboard_hourly_monthly.parquet",
        "robust":    "dashboard_robustness.parquet",
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            data[key] = pd.read_parquet(path)
        else:
            print(f"Fichier manquant : {path}")
            data[key] = pd.DataFrame()

    kpi_path = os.path.join(DATA_DIR, "dashboard_metrics.json")
    if os.path.exists(kpi_path):
        with open(kpi_path, encoding="utf-8") as f:
            data["kpis"] = json.load(f)
    else:
        data["kpis"] = {}

    return data


# Chargement initial
_data        = load_data()
df_main      = _data.get("main",      pd.DataFrame())
df_daily     = _data.get("daily",     pd.DataFrame())
df_profile   = _data.get("profile",   pd.DataFrame())
df_profile_m = _data.get("profile_m", pd.DataFrame())
df_robust    = _data.get("robust",    pd.DataFrame())
kpis         = _data.get("kpis",      {})

# Typage datetime
if not df_main.empty:
    df_main["datetime"] = pd.to_datetime(df_main["datetime"])
if not df_daily.empty:
    df_daily["date"] = pd.to_datetime(df_daily["date"])


# ─────────────────────────────────────────────────────────────
# HELPER : WRAPPER LOADING SPINNER
# ─────────────────────────────────────────────────────────────

def with_loading(component):
    """Entoure un composant d'un spinner de chargement."""
    return dcc.Loading(
        type="circle",
        color=COLORS["primary"],
        children=component,
    )


# ─────────────────────────────────────────────────────────────
# COMPOSANT KPI CARD
# ─────────────────────────────────────────────────────────────

def kpi_card(title, value, subtitle="", color=COLORS["primary"], icon=""):
    """Card KPI avec icone, valeur coloree et sous-titre."""
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={"fontSize": "1.8rem"}),
                html.Div([
                    html.H4(
                        value, className="mb-0",
                        style={"color": color, "fontWeight": "800", "fontSize": "1.5rem"},
                    ),
                    html.P(
                        title, className="mb-0",
                        style={"color": COLORS["text"], "fontWeight": "600", "fontSize": "0.85rem"},
                    ),
                    html.P(
                        subtitle, className="mb-0",
                        style={"color": COLORS["subtext"], "fontSize": "0.75rem"},
                    ),
                ], style={"marginLeft": "12px"}),
            ], style={"display": "flex", "alignItems": "center"}),
        ]),
        style={
            "backgroundColor": COLORS["card_bg"],
            "border":          f"1px solid {hex_to_rgba(color, 0.35)}",
            "borderRadius":    "12px",
            "boxShadow":       f"0 4px 15px {hex_to_rgba(color, 0.15)}",
        },
        className="mb-2",
    )


def build_kpi_row():
    """Ligne de 5 KPI cards en haut du dashboard."""
    return dbc.Row([
        dbc.Col(kpi_card(
            "MAE XGBoost",
            f"{kpis.get('xgb_mae', '—')} t/h",
            subtitle="Jeu de test",
            color=COLORS["success"], icon="🎯",
        ), md=2),
        dbc.Col(kpi_card(
            "sMAPE XGBoost",
            f"{kpis.get('xgb_smape', '—')}%",
            subtitle="Erreur relative",
            color=COLORS["primary"], icon="📉",
        ), md=2),
        dbc.Col(kpi_card(
            "Gain vs Baseline",
            f"-{kpis.get('improvement_mae_pct', '—')}%",
            subtitle="Reduction MAE",
            color=COLORS["warning"], icon="🚀",
        ), md=2),
        dbc.Col(kpi_card(
            "Trajets totaux",
            f"{kpis.get('total_rides_all', 0):,}",
            subtitle="Capital Bikeshare DC 2024",
            color=COLORS["secondary"], icon="🚲",
        ), md=3),
        dbc.Col(kpi_card(
            "Demande moy/h",
            f"{kpis.get('avg_hourly_demand', '—')}",
            subtitle=f"Pic : {kpis.get('peak_demand', '—')} t/h",
            color=COLORS["danger"], icon="⏱️",
        ), md=3),
    ], className="mb-3")


# ─────────────────────────────────────────────────────────────
# FIGURES — ONGLET 1 : VUE GENERALE
# ─────────────────────────────────────────────────────────────

def fig_serie_temporelle():
    """Serie temporelle journaliere avec moyennes mobiles."""
    if df_daily.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_daily["date"], y=df_daily["total_rides"],
        name="Journalier",
        marker_color=COLORS["real"], marker_opacity=0.35,
    ))
    fig.add_trace(go.Scatter(
        x=df_daily["date"],
        y=df_daily["total_rides"].rolling(7, center=True).mean(),
        name="Moy. 7j",
        line=dict(color=COLORS["primary"], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=df_daily["date"],
        y=df_daily["total_rides"].rolling(28, center=True).mean(),
        name="Moy. 28j",
        line=dict(color=COLORS["warning"], width=2, dash="dot"),
    ))
    fig.update_layout(
        title="Demande journaliere — Capital Bikeshare 2024",
        xaxis_title="Date", yaxis_title="Trajets / jour",
        hovermode="x unified",
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


def fig_profil_horaire():
    """Profil horaire moyen avec intervalle de confiance 95%."""
    if df_profile.empty:
        return go.Figure()

    fig = go.Figure()
    style_map = {
        "Semaine":    dict(color=COLORS["primary"],   dash="solid", width=2.5),
        "Week-end":   dict(color=COLORS["secondary"], dash="dot",   width=2.5),
        "Jour ferie": dict(color=COLORS["warning"],   dash="dash",  width=2),
    }
    # Accepter plusieurs variantes d'orthographe dans les donnees
    aliases = {
        "Semaine":    ["Semaine"],
        "Week-end":   ["Week-end"],
        "Jour ferie": ["Jour ferie", "Jour ferie", "Jour ferie"],
    }

    for display, style in style_map.items():
        sub = df_profile[df_profile["day_type"].isin(aliases[display])]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["hour"], y=sub["mean_rides"],
            name=display, mode="lines+markers",
            line=style, marker=dict(size=5),
        ))
        if "ci_low" in sub.columns and "ci_high" in sub.columns:
            x_band = pd.concat([sub["hour"], sub["hour"].iloc[::-1]])
            y_band = pd.concat([sub["ci_high"], sub["ci_low"].iloc[::-1]])
            fig.add_trace(go.Scatter(
                x=x_band, y=y_band,
                fill="toself",
                fillcolor=hex_to_rgba(style["color"], 0.12),
                line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))

    layout = PLOTLY_TEMPLATE["layout"].copy()

    layout["xaxis"] = {
        **layout.get("xaxis", {}),
        "tickvals": list(range(24)),
        "title": "Heure",
        }

    layout["yaxis"] = {
        **layout.get("yaxis", {}),
        "title": "Trajets moyens / heure",
        }

    fig.update_layout(
        title="Profil horaire moyen — IC 95%",
        **layout,
        )

    return fig


def fig_member_casual():
    """Repartition membres vs casuals par mois."""
    if df_daily.empty:
        return go.Figure()

    monthly = (
        df_daily.groupby("month")[["member_rides", "casual_rides"]]
        .sum()
        .reset_index()
    )
    month_names = ["Jan", "Fev", "Mar", "Avr", "Mai", "Jun",
                   "Jul", "Aou", "Sep", "Oct", "Nov", "Dec"]
    monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m - 1])

    total = monthly["member_rides"] + monthly["casual_rides"]
    pct_casual = 100 * monthly["casual_rides"] / total.replace(0, np.nan)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Trajets par mois (membres vs casuals)",
                        "% Casuals — saisonnalite touristique"],
    )
    fig.add_trace(go.Bar(
        x=monthly["month_name"], y=monthly["member_rides"],
        name="Membres", marker_color=COLORS["primary"], opacity=0.85,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=monthly["month_name"], y=monthly["casual_rides"],
        name="Casuals", marker_color=COLORS["secondary"], opacity=0.85,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=monthly["month_name"], y=pct_casual,
        name="% Casuals", mode="lines+markers",
        line=dict(color=COLORS["warning"], width=2.5),
        fill="tozeroy",
        fillcolor=hex_to_rgba(COLORS["warning"], 0.12),
    ), row=1, col=2)

    fig.update_layout(
        barmode="stack", height=380,
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


# ─────────────────────────────────────────────────────────────
# FIGURES — ONGLET 2 : PREDICTIONS
# ─────────────────────────────────────────────────────────────

def fig_predictions_vs_real(n_days=30):
    """Graphique principal predictions vs reel + MAE glissante."""
    if df_main.empty:
        return go.Figure()

    n   = min(n_days * 24, len(df_main))
    sub = df_main.iloc[:n].copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=["Prediction vs Reel (trajets/heure)",
                        "MAE glissante 24h"],
    )

    # Reel
    fig.add_trace(go.Scatter(
        x=sub["datetime"], y=sub["total_rides"],
        name="Reel",
        line=dict(color=COLORS["real"], width=1.2), opacity=0.9,
    ), row=1, col=1)

    # XGBoost
    fig.add_trace(go.Scatter(
        x=sub["datetime"], y=sub["pred_xgb"],
        name="XGBoost",
        line=dict(color=COLORS["xgb"], width=1.5, dash="dot"),
    ), row=1, col=1)

    # Naif saisonnier
    if "pred_naive" in sub.columns:
        fig.add_trace(go.Scatter(
            x=sub["datetime"], y=sub["pred_naive"],
            name="Naif saisonnier",
            line=dict(color=COLORS["naive"], width=1, dash="dash"),
            opacity=0.6,
        ), row=1, col=1)

    # MAE glissante 24h
    if "err_xgb" in sub.columns:
        mae_roll = sub["err_xgb"].rolling(24, min_periods=1).mean()
    elif "mae_rolling_24h" in sub.columns:
        mae_roll = sub["mae_rolling_24h"]
    else:
        mae_roll = pd.Series(dtype=float)

    if not mae_roll.empty:
        fig.add_trace(go.Scatter(
            x=sub["datetime"], y=mae_roll,
            name="MAE 24h",
            fill="tozeroy",
            fillcolor=hex_to_rgba(COLORS["danger"], 0.18),
            line=dict(color=COLORS["danger"], width=1.5),
        ), row=2, col=1)

    fig.update_layout(
        hovermode="x unified", height=520,
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


def fig_heatmap_dow_hour():
    """Heatmap demande moyenne : jour de semaine x heure."""
    if df_main.empty:
        return go.Figure()

    pivot = (
        df_main.groupby(["dow", "hour"])["total_rides"]
        .mean()
        .reset_index()
        .pivot(index="dow", columns="hour", values="total_rides")
    )
    day_labels = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h}h" for h in pivot.columns],
        y=[day_labels[i] for i in pivot.index],
        colorscale="YlOrRd",
        colorbar=dict(title="Trajets/h"),
        hovertemplate="%{y} a %{x}<br>Moy : %{z:.0f} trajets<extra></extra>",
    ))
    fig.update_layout(
        title="Heatmap demande — Jour x Heure",
        xaxis_title="Heure", height=320,
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


def fig_distribution_erreurs():
    """Distribution des erreurs absolues : XGBoost vs Naif."""
    if df_main.empty:
        return go.Figure()

    fig = go.Figure()
    for col, name, color in [
        ("err_naive", "Naif saisonnier", COLORS["naive"]),
        ("err_xgb",   "XGBoost",         COLORS["xgb"]),
    ]:
        if col not in df_main.columns:
            continue
        vals = df_main[col].dropna()
        fig.add_trace(go.Histogram(
            x=vals, name=name, nbinsx=60,
            marker_color=color, opacity=0.6,
            histnorm="probability density",
        ))

    fig.update_layout(
        title="Distribution des erreurs absolues (densite)",
        xaxis_title="Erreur absolue (trajets/h)",
        yaxis_title="Densite",
        barmode="overlay",
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


# ─────────────────────────────────────────────────────────────
# FIGURES — ONGLET 3 : ROBUSTESSE
# ─────────────────────────────────────────────────────────────

def fig_robustesse():
    """Barres horizontales : MAE par segment + gain XGBoost vs Naif."""
    if df_robust.empty:
        return go.Figure()

    df_r = df_robust.sort_values("MAE_xgb", ascending=True)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["MAE par segment (trajets/h)",
                        "Gain XGBoost vs Naif (%)"],
    )

    # MAE comparee
    fig.add_trace(go.Bar(
        x=df_r["MAE_naive"], y=df_r["segment"],
        name="Naif saisonnier", orientation="h",
        marker_color=COLORS["naive"], opacity=0.7,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df_r["MAE_xgb"], y=df_r["segment"],
        name="XGBoost", orientation="h",
        marker_color=COLORS["xgb"], opacity=0.85,
    ), row=1, col=1)

    # Gain %
    gain_colors = [
        COLORS["success"] if g > 0 else COLORS["danger"]
        for g in df_r["gain_pct"]
    ]
    fig.add_trace(go.Bar(
        x=df_r["gain_pct"], y=df_r["segment"],
        name="Gain %", orientation="h",
        marker_color=gain_colors, showlegend=False,
        text=[f"{g:+.1f}%" for g in df_r["gain_pct"]],
        textposition="outside",
    ), row=1, col=2)
    fig.add_vline(x=0, line_color="white", line_dash="dash", row=1, col=2)

    fig.update_layout(
        height=520, barmode="group",
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


def _build_robustness_table():
    """Tableau HTML des metriques de robustesse par segment."""
    if df_robust.empty:
        return html.P("Donnees non disponibles",
                      style={"color": COLORS["subtext"]})

    cols = ["segment", "n", "mean_demand", "MAE_naive", "MAE_xgb", "sMAPE_xgb", "gain_pct"]
    cols = [c for c in cols if c in df_robust.columns]
    df_r = df_robust[cols].sort_values("MAE_xgb")

    header = html.Tr([
        html.Th(c, style={
            "padding": "8px 12px",
            "color": COLORS["subtext"],
            "fontSize": "0.8rem",
            "borderBottom": f"1px solid {COLORS['grid']}",
        }) for c in cols
    ])

    rows = []
    for _, row in df_r.iterrows():
        gain_val   = row.get("gain_pct", 0)
        gain_color = COLORS["success"] if gain_val > 0 else COLORS["danger"]
        cells = []
        for c in cols:
            v = row[c]
            if c == "gain_pct":
                text       = f"{v:+.1f}%"
                cell_style = {"color": gain_color, "fontWeight": "700"}
            elif isinstance(v, float):
                text       = f"{v:.1f}"
                cell_style = {"color": COLORS["text"]}
            else:
                text       = str(v)
                cell_style = {"color": COLORS["text"]}
            cells.append(html.Td(text, style={
                **cell_style,
                "padding":    "6px 12px",
                "fontSize":   "0.82rem",
                "borderBottom": f"1px solid {COLORS['grid']}",
            }))
        rows.append(html.Tr(cells))

    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={
            "width":           "100%",
            "borderCollapse":  "collapse",
            "backgroundColor": COLORS["card_bg"],
            "borderRadius":    "8px",
            "overflow":        "hidden",
        },
    )


# ─────────────────────────────────────────────────────────────
# FIGURES — ONGLET 4 : ANALYSE EDA
# ─────────────────────────────────────────────────────────────

def fig_heatmap_monthly():
    """Heatmap demande moyenne : mois x heure."""
    if df_profile_m.empty:
        return go.Figure()

    pivot = df_profile_m.pivot(index="month", columns="hour", values="mean_rides")
    month_names = ["Jan", "Fev", "Mar", "Avr", "Mai", "Jun",
                   "Jul", "Aou", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h}h" for h in pivot.columns],
        y=[month_names[i - 1] for i in pivot.index],
        colorscale="Viridis",
        colorbar=dict(title="Trajets/h"),
        hovertemplate="Mois %{y} a %{x}<br>Moy : %{z:.0f} trajets<extra></extra>",
    ))
    fig.update_layout(
        title="Heatmap demande — Mois x Heure",
        xaxis_title="Heure", height=350,
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


def _build_stats_cards():
    """4 cards de statistiques descriptives."""
    stats = [
        ("% Membres",     f"{kpis.get('pct_member', '—')}%",             COLORS["primary"]),
        ("% Casuals",     f"{kpis.get('pct_casual', '—')}%",             COLORS["secondary"]),
        ("Pic max obs.",  f"{kpis.get('peak_demand', '—')} t/h",         COLORS["danger"]),
        ("Seuil pic 90%", f"{kpis.get('peak_hour_threshold', '—')} t/h", COLORS["warning"]),
    ]
    return dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4(v, style={"color": c, "fontWeight": "800"}),
                    html.P(t, style={
                        "color": COLORS["subtext"],
                        "fontSize": "0.85rem",
                        "marginBottom": 0,
                    }),
                ]),
                style={
                    "backgroundColor": COLORS["card_bg"],
                    "border":          f"1px solid {hex_to_rgba(c, 0.3)}",
                    "borderRadius":    "10px",
                    "textAlign":       "center",
                },
            ),
            md=3, className="mb-3",
        )
        for t, v, c in stats
    ])


def fig_scatter_pred_vs_real():
    """Scatter : valeurs predites vs valeurs reelles (XGBoost)."""
    if df_main.empty:
        return go.Figure()

    sample = df_main.sample(min(3000, len(df_main)), random_state=42)
    max_val = max(sample["total_rides"].max(), sample["pred_xgb"].max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample["total_rides"], y=sample["pred_xgb"],
        mode="markers",
        marker=dict(color=COLORS["xgb"], size=4, opacity=0.35),
        name="XGBoost",
        hovertemplate="Reel : %{x:.0f}<br>Predit : %{y:.0f}<extra></extra>",
    ))
    # Droite parfaite y = x
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color="white", width=1, dash="dash"),
        name="Prediction parfaite",
        showlegend=True,
    ))
    fig.update_layout(
        title="Predit vs Reel — XGBoost (echantillon 3000 pts)",
        xaxis_title="Demande reelle (trajets/h)",
        yaxis_title="Demande predite (trajets/h)",
        height=350,
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


# ─────────────────────────────────────────────────────────────
# LAYOUT COMPLET
# ─────────────────────────────────────────────────────────────

app.layout = dbc.Container(fluid=True, style={
    "backgroundColor": COLORS["dark"],
    "minHeight": "100vh",
    "padding": "0 24px",
}, children=[

    # ── Header ──────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.H1(
                "Capital Bikeshare — Demand Forecasting",
                style={
                    "color": COLORS["text"],
                    "fontWeight": "800",
                    "fontSize": "1.7rem",
                    "marginBottom": "2px",
                },
            ),
            html.P(
                "Prediction horaire J+2 · XGBoost · Washington D.C. · 2024",
                style={"color": COLORS["subtext"], "fontSize": "0.85rem"},
            ),
        ], md=8),
        dbc.Col([
            html.Div(
                f"Periode : {kpis.get('period_start', '')}  ->  {kpis.get('period_end', '')}",
                style={
                    "color":     COLORS["subtext"],
                    "fontSize":  "0.82rem",
                    "textAlign": "right",
                    "paddingTop":"14px",
                },
            ),
        ], md=4),
    ], className="py-3 mb-2",
       style={"borderBottom": f"1px solid {COLORS['grid']}"}),

    # ── KPIs ────────────────────────────────────────────────
    build_kpi_row(),

    # ── Onglets ─────────────────────────────────────────────
    dbc.Tabs(id="main-tabs", active_tab="tab-overview",
             style={"borderBottom": f"1px solid {COLORS['grid']}"}, children=[

        # ═══════════════════════════════════════════
        # ONGLET 1 — Vue Generale
        # ═══════════════════════════════════════════
        dbc.Tab(label="Vue Generale", tab_id="tab-overview", children=[

            dbc.Row(className="mt-3", children=[
                dbc.Col([
                    with_loading(dcc.Graph(
                        id="graph-serie",
                        figure=fig_serie_temporelle(),
                        config={"displayModeBar": False},
                        style={"height": "320px"},
                    ))
                ], md=12),
            ]),

            dbc.Row([
                dbc.Col([
                    with_loading(dcc.Graph(
                        id="graph-profil-horaire",
                        figure=fig_profil_horaire(),
                        config={"displayModeBar": False},
                        style={"height": "360px"},
                    ))
                ], md=6),
                dbc.Col([
                    with_loading(dcc.Graph(
                        id="graph-member-casual",
                        figure=fig_member_casual(),
                        config={"displayModeBar": False},
                        style={"height": "360px"},
                    ))
                ], md=6),
            ]),
        ]),

        # ═══════════════════════════════════════════
        # ONGLET 2 — Predictions
        # ═══════════════════════════════════════════
        dbc.Tab(label="Predictions", tab_id="tab-pred", children=[

            dbc.Row(className="mt-3", children=[
                dbc.Col([
                    html.Label(
                        "Fenetre d'affichage (jours) :",
                        style={"color": COLORS["text"], "fontSize": "0.85rem"},
                    ),
                    dcc.Slider(
                        id="slider-days",
                        min=7, max=60, step=7, value=30,
                        marks={d: str(d) for d in [7, 14, 21, 30, 45, 60]},
                        tooltip={"placement": "bottom"},
                    ),
                ], md=6),
            ]),

            dbc.Row([
                dbc.Col([
                    with_loading(dcc.Graph(
                        id="graph-predictions",
                        figure=fig_predictions_vs_real(30),
                        config={"displayModeBar": True, "scrollZoom": True},
                        style={"height": "520px"},
                    ))
                ], md=12),
            ]),

            dbc.Row(className="mt-2", children=[
                dbc.Col([
                    with_loading(dcc.Graph(
                        id="graph-heatmap",
                        figure=fig_heatmap_dow_hour(),
                        config={"displayModeBar": False},
                        style={"height": "320px"},
                    ))
                ], md=6),
                dbc.Col([
                    with_loading(dcc.Graph(
                        id="graph-error-dist",
                        figure=fig_distribution_erreurs(),
                        config={"displayModeBar": False},
                        style={"height": "320px"},
                    ))
                ], md=6),
            ]),
        ]),

        # ═══════════════════════════════════════════
        # ONGLET 3 — Robustesse
        # ═══════════════════════════════════════════
        dbc.Tab(label="Robustesse", tab_id="tab-robust", children=[

            dbc.Row(className="mt-3", children=[
                dbc.Col([
                    html.H5("Analyse de robustesse par segment",
                            style={"color": COLORS["text"]}),
                    html.P(
                        "Performance du modele selon le type de jour, la plage horaire, "
                        "la meteo et le niveau de demande. "
                        "Gain positif = XGBoost bat le naif sur ce segment.",
                        style={"color": COLORS["subtext"], "fontSize": "0.85rem"},
                    ),
                ], md=12),
            ]),

            dbc.Row([
                dbc.Col([
                    with_loading(dcc.Graph(
                        id="graph-robustesse",
                        figure=fig_robustesse(),
                        config={"displayModeBar": False},
                        style={"height": "520px"},
                    ))
                ], md=12),
            ]),

            dbc.Row([
                dbc.Col([
                    html.H6("Tableau detaille",
                            style={"color": COLORS["text"], "marginTop": "12px"}),
                    html.Div(
                        id="table-robustesse",
                        children=_build_robustness_table(),
                        style={"overflowX": "auto"},
                    ),
                ], md=12),
            ]),
        ]),

        # ═══════════════════════════════════════════
        # ONGLET 4 — Analyse EDA
        # ═══════════════════════════════════════════
        dbc.Tab(label="Analyse EDA", tab_id="tab-eda", children=[

            dbc.Row(className="mt-3", children=[
                dbc.Col([
                    with_loading(dcc.Graph(
                        id="graph-heatmap-full",
                        figure=fig_heatmap_dow_hour(),
                        config={"displayModeBar": False},
                        style={"height": "340px"},
                    ))
                ], md=6),
                dbc.Col([
                    with_loading(dcc.Graph(
                        id="graph-heatmap-monthly",
                        figure=fig_heatmap_monthly(),
                        config={"displayModeBar": False},
                        style={"height": "340px"},
                    ))
                ], md=6),
            ]),

            dbc.Row([
                dbc.Col([
                    with_loading(dcc.Graph(
                        id="graph-scatter",
                        figure=fig_scatter_pred_vs_real(),
                        config={"displayModeBar": False},
                        style={"height": "350px"},
                    ))
                ], md=6),
                dbc.Col([
                    html.H5("Statistiques cles",
                            style={"color": COLORS["text"], "marginTop": "10px"}),
                    _build_stats_cards(),
                ], md=6),
            ], className="mt-2"),
        ]),

    ]),

    # ── Footer ──────────────────────────────────────────────
    html.Hr(style={"borderColor": COLORS["grid"], "marginTop": "30px"}),
    html.P(
        "Capital Bikeshare Demand Forecasting POV  ·  "
        "XGBoost  ·  Split temporel strict  ·  Python",
        style={
            "color":     COLORS["subtext"],
            "fontSize":  "0.75rem",
            "textAlign": "center",
            "marginBottom": "16px",
        },
    ),
])


# ─────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("graph-predictions", "figure"),
    Input("slider-days", "value"),
)
def update_predictions(n_days):
    """Met a jour le graphique predictions selon la fenetre choisie."""
    return fig_predictions_vs_real(n_days=n_days)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nDashboard Capital Bikeshare")
    print("=" * 45)
    if not df_main.empty:
        print(f"  Donnees chargees : {len(df_main):,} heures de test")
        print(f"  MAE XGBoost = {kpis.get('xgb_mae')}  |  "
              f"sMAPE = {kpis.get('xgb_smape')}%")
        print(f"  Gain vs baseline : -{kpis.get('improvement_mae_pct')}%")
    else:
        print("  ATTENTION : donnees manquantes")
        print("  -> Executer le notebook 03_Dashboard_Prep.ipynb d'abord")
    print("\n  Ouvrir : http://127.0.0.1:8050\n")
    app.run(debug=True, host="127.0.0.1", port=8050)