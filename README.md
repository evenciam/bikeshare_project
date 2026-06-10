# 🚲 Capital Bikeshare — Demand Forecasting POV

> **Objectif** : Prédire la demande horaire de vélos en libre-service (J+1 et J+2) pour aider un opérateur de mobilité à optimiser la redistribution de sa flotte.

---

## 📁 Structure du projet

```
bikeshare_project/
│
├── data/
│   ├── raw/                    # Données brutes téléchargées (non versionnées)
│   └── processed/              # Données nettoyées et features engineered
│
├── src/
│   ├── 01_data_download.py     # Téléchargement & concaténation des données
│   ├── 02_eda.py               # Analyse exploratoire
│   ├── 03_feature_engineering.py  # Construction des features
│   ├── 04_baseline_model.py    # Modèle baseline (moyenne glissante / SARIMA)
│   ├── 05_advanced_model.py    # Modèle avancé (XGBoost + features riches)
│   ├── 06_evaluation.py        # Métriques & analyse de robustesse
│   └── utils.py                # Fonctions utilitaires partagées
│
├── notebooks/
│   └── 03_Dashboard_Prep.ipynb # Préparation des données pour le dashboard
│
├── dashboard/
│   └── app.py                  # Dashboard Plotly Dash interactif
│
├── docs/
│   └── NOTE_METHODE.md         # Note méthodologique (< 1 page)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Installation & Exécution

### 1. Cloner le dépôt

```bash
git clone https://github.com/evenciam/bikeshare_project.git
cd bikeshare_project
```

### 2. Créer l'environnement virtuel

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Télécharger les données

```bash
python src/01_data_download.py
```

> Les données sont téléchargées depuis https://capitalbikeshare.com/system-data  
> et sauvegardées dans `data/raw/`. Ce dossier est exclu du versioning (`.gitignore`).

### 5. Lancer le pipeline complet

```bash
python src/02_eda.py
python src/03_feature_engineering.py
python src/04_baseline_model.py
python src/05_advanced_model.py
python src/06_evaluation.py
```

### 6. Lancer le dashboard

```bash
python dashboard/app.py
# Ouvrir  http://127.0.0.1:8050
```

### 7. Jupyter Notebooks (optionnel)

```bash
jupyter lab
```

---

## 📊 Résultats des modèles

| Modèle | MAE | sMAPE | RMSE |
|--------|-----|-------|------|
| Baseline (moyenne glissante 7j) | ~131.64 trajets/h | ~17.31% | ~246.45 |
| XGBoost + Feature Engineering | ~142.17 trajets/h | ~28.2% | ~221.01 |

> *Résultats sur le jeu de test temporel (données post-octobre 2025)*

---

## 🔑 Points clés

- **Split temporel strict** : pas de data leakage (train < validation < test dans l'ordre chronologique)
- **Features météo** : intégrées via Open-Meteo API (gratuit, sans clé)
- **Analyse de robustesse** : comportement sur pics, jours fériés, météo extrême
- **Dashboard interactif** : visualisation des prédictions vs réel, analyse des erreurs

---

## 👤 Auteur

[MICHONDARD Evencia] — Master Mathématiques Appliquées  
[[LinkedIn](https://www.linkedin.com/in/evencia-michondard)] | [[GitHub](https://github.com/evenciam)]
