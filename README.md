
---

# Store Item Forecasting & Inventory Optimization

##  Description du Projet

Ce projet effectue la **prévision des ventes quotidiennes** pour différents magasins et articles, et propose des **recommandations de stock optimisé**.
Il combine un modèle de machine learning (**ExtraTreesRegressor**) avec des méthodes de gestion des stocks pour minimiser les coûts tout en respectant un niveau de service désiré.

---

##  Objectifs du Projet

* Préparer et enrichir les données avec des variables temporelles et des lags de ventes
* Prédire les ventes futures pour chaque magasin et article
* Calculer le stock de sécurité et le niveau de stock recommandé
* Estimer les coûts de stockage et de rupture de stock
* Générer des fichiers de sortie opérationnels pour la planification des stocks

---

##  Approche

1. Chargement et exploration des données (`train.csv`, `test.csv`, `sample_submission.csv`)
2. Prétraitement : conversion des dates, création des colonnes year/month/day/weekday
3. Création de features de lag (1, 7 et 30 jours) pour capturer la tendance passée
4. Modélisation des ventes avec **ExtraTreesRegressor**
5. Évaluation du modèle sur un jeu de validation (RMSE, MAE)
6. Génération de recommandations de stock :

   * Calcul du stock de sécurité basé sur l’erreur de prédiction et le niveau de service
   * Calcul du stock recommandé et estimation des coûts associés
7. Sauvegarde des résultats dans `submission.csv`, `inventory_plan.csv`, et `submission_optimized.csv`

---

##  Résultats

* Prévision des ventes pour chaque magasin et article
* Calcul du stock recommandé avec prise en compte des risques de rupture et coûts de stockage
* Fichiers prêts pour l’intégration dans un plan opérationnel

---

##  Fichiers Principaux

* `store_item_forecast_and_inventory_opt.py` — Script complet de traitement, prévision et optimisation des stocks
* `train.csv`, `test.csv`, `sample_submission.csv` — Datasets utilisés
* `submission.csv`, `inventory_plan.csv`, `submission_optimized.csv` — Fichiers générés
* `README.md` — Documentation

---

##  Auteur

Projet réalisé par **Chahboune Ismail**

---

