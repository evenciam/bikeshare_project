# NOTE MÉTHODOLOGIQUE — Capital Bikeshare Demand Forecasting

## 1. Définition du problème & horizon de prédiction

**Cas d'usage** : Prédire la demande horaire de vélos à **J+1 et J+2** (24h et 48h à l'avance).

**Justification opérationnelle** : Un opérateur de mobilité doit anticiper les déséquilibres de flotte *avant* qu'ils surviennent. Une prévision à 24-48h permet de planifier les opérations de rebalancing nocturne (déplacement de vélos entre stations) et d'allouer les équipes de terrain. Un horizon plus court (< 6h) ne laisse pas le temps d'agir ; un horizon plus long (> 72h) dégrade trop la précision.

**Granularité** : Agrégation au niveau **ville entière, par heure**, ce qui est l'indicateur clé pour le dimensionnement de la flotte globale. Une extension par station serait l'étape suivante.

---

## 2. Stratégie de validation — Split temporel strict

```
|-------- TRAIN (Jan 2024 → Sep 2024) --------|-- VAL (Oct 2024) --|-- TEST (Nov-Dec 2024) --|
```

**Règle absolue** : aucune donnée future ne filtre dans le passé.
- Les features lag (valeurs passées) sont construites **uniquement** à partir de données antérieures à la fenêtre de prédiction.
- Les moyennes mobiles (rolling means) utilisent `.shift(48)` minimum pour éviter tout leakage sur un horizon J+2.
- Validation sur octobre (mois de transition saisonnière, cas difficile volontaire).

---

## 3. Baseline vs Amélioration

| Critère | Baseline (Moyenne glissante 7j) | XGBoost + Features |
|---|---|---|
| MAE | ~45 trajets/h | ~18 trajets/h |
| sMAPE | ~38% | ~16% |
| Pics (top 5%) | Sous-estimés de 60% | Sous-estimés de 25% |
| Jours fériés | Non capturés | Partiellement capturés |
| Temps d'entraînement | < 1s | ~30s |

**Gains clés du feature engineering** : lags temporels (48h, 168h), variables cycliques (heure sin/cos), météo (température, précipitations), indicateur jour férié.

---

## 4. Limites identifiées

- **Pics imprévus** (événements sportifs, concerts) : le modèle sous-estime systématiquement car aucune donnée d'événement n'est intégrée.
- **Données météo** : dépendance à une API externe ; en production, une météo prévisionnelle (et non observée) doit être utilisée, introduisant une incertitude supplémentaire.
- **Granularité station** : le modèle agrégé ne capture pas les déséquilibres locaux.
- **Concept drift** : changement de comportement post-COVID ou lié à l'expansion du réseau.

---

## 5. Vision MLOps — Industrialisation

**Orchestration** : Apache Airflow ou Prefect — DAG quotidien déclenché à J-1 minuit.

**Pipeline** :
1. Ingestion données temps réel (API Capital Bikeshare)
2. Récupération météo prévisionnelle (Open-Meteo)
3. Feature engineering → scoring du modèle
4. Écriture des prédictions en base (PostgreSQL/BigQuery)
5. Alertes si sMAPE > seuil sur fenêtre glissante

**Monitoring** : suivi de la distribution des features (data drift via Evidently AI), métriques de performance hebdomadaires, ré-entraînement mensuel ou déclenché par dégradation > 20% MAE.

**Stockage modèle** : MLflow pour le versioning et la comparaison des runs.