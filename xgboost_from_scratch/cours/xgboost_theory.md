# 📘 XGBoost : Théorie et Implémentation From Scratch

---

## 🎯 Objectif du cours

Ce cours a pour but de comprendre en profondeur le fonctionnement de l’algorithme XGBoost, depuis les **fondations mathématiques** jusqu’à son **implémentation complète à la main**. L’objectif est d'en saisir les **concepts clés**, les **formules d'optimisation**, et de développer une version simplifiée de XGBoost.

---

## 📌 1. Introduction à XGBoost

**XGBoost (Extreme Gradient Boosting)** est une optimisation du **Gradient Boosting** :
- Plus rapide (grâce à la parallélisation et au calcul distribué),
- Plus régularisé (meilleure gestion du surapprentissage),
- Plus performant sur des tâches tabulaires (Kaggle, production).

---

## 🧠 2. Concepts clés

### 🔸 2.1 Boosting
Le **boosting** est une méthode d’**apprentissage itératif** :
- On ajoute à chaque itération un modèle (souvent un arbre) pour corriger les erreurs du précédent.
- Chaque modèle apprend sur les **résidus (erreurs)** des prédictions précédentes.

### 🔸 2.2 Gradient Boosting
Au lieu de corriger directement les erreurs, on entraîne les modèles sur le **gradient négatif** de la fonction de perte. C’est une **descente de gradient fonctionnelle**.

### 🔸 2.3 Arbres de décision comme learners faibles
XGBoost utilise des **arbres CART** (Classification And Regression Trees) comme modèles faibles.

---

## 📐 3. Formulation mathématique de XGBoost

On cherche à **minimiser** la fonction :

\[
\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)
\]

- \( l \) : fonction de perte (ex. MSE ou log loss)
- \( f_t \) : le nouvel arbre à apprendre à l'étape \( t \)
- \( \Omega(f_t) \) : terme de régularisation pour éviter l’overfitting :

\[
\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2
\]

avec :
- \( T \) : nombre de feuilles
- \( w_j \) : score assigné à la feuille \( j \)
- \( \gamma, \lambda \) : paramètres de régularisation

---

### 🔧 3.1 Approximation par second ordre (Taylor expansion)

Pour faciliter l’optimisation, on fait une expansion de Taylor à l’ordre 2 autour de \( \hat{y}_i^{(t-1)} \) :

\[
\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2 \right] + \Omega(f_t)
\]

où :
- \( g_i = \frac{\partial l(y_i, \hat{y}_i)}{\partial \hat{y}_i} \) : **gradient**
- \( h_i = \frac{\partial^2 l(y_i, \hat{y}_i)}{\partial \hat{y}_i^2} \) : **hessienne**

---

### 🌳 3.2 Construction de l'arbre

Soit un arbre avec \( T \) feuilles. On cherche les poids optimaux \( w_j \) pour chaque feuille :

\[
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
\]

\[
\text{Gain (amélioration)} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma
\]

- \( I_L, I_R \) : indices des points dans les sous-feuilles gauche/droite
- On **choisit le split qui maximise le gain**.

---

## 🧩 4. Algorithme étape par étape

1. Initialiser les prédictions \( \hat{y}_i^{(0)} \) à une constante (ex : moyenne pour MSE).
2. Pour chaque étape \( t \in [1, T] \) :
   - Calculer les gradients \( g_i \) et hessiennes \( h_i \)
   - Construire un arbre \( f_t \) qui minimise l'objectif approximé
   - Mettre à jour les prédictions :  
     \[
     \hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)
     \]
3. Répéter jusqu’à convergence ou max de rounds.

---

## 🛠️ 5. Vers l’implémentation from scratch

Pour l’implémentation dans `xgboost_scratch.py`, voici les étapes que nous allons suivre :

### ✅ Étapes de l’implémentation
- [ ] Définir une classe `XGBoostRegressor`
- [ ] Implémenter la loss (ex: MSE → gradients et hessiennes)
- [ ] Créer un arbre binaire récursif basé sur le gain
- [ ] Implémenter la prédiction arbre par arbre
- [ ] Ajouter régularisation \( \lambda \), \( \gamma \)
- [ ] Implémenter l'ensemble du boosting
- [ ] Ajouter early stopping (facultatif)

---

## 📎 6. Remarques

- XGBoost repose sur une approximation mathématique rigoureuse.
- Sa force vient de son **contrôle fin du surapprentissage**, sa **vitesse** et ses **optimisations bas niveau** (non reproduites ici).
- Le but ici est **l’apprentissage, pas la performance**.

---

## ✏️ Références

- [Chen & Guestrin, 2016 - XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [XGBoost documentation officielle](https://xgboost.readthedocs.io/)
- [XGBoost GitHub](https://github.com/dmlc/xgboost)