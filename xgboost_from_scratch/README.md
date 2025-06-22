# 📘 Théorie XGBoost — Formules et Optimisation

---

## 🎯 Fonction objectif de XGBoost

À chaque étape \( t \), XGBoost cherche à minimiser une fonction objectif composée de :

- une **fonction de perte** \( l \),
- un **terme de régularisation** \( \Omega \).

La fonction totale s’écrit :

$$
\mathcal{L}^{(t)} = \sum_{i=1}^n l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)
$$

où :
- \( \hat{y}_i^{(t-1)} \) est la prédiction de l'étape précédente,
- \( f_t(x_i) \) est le nouvel arbre à apprendre,
- \( \Omega(f_t) \) pénalise la complexité du modèle.

---

## 🧱 Terme de régularisation

Le terme de régularisation contrôle le **nombre de feuilles** et les **valeurs attribuées aux feuilles** :

$$
\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2
$$

avec :
- \( T \) : nombre de feuilles de l’arbre,
- \( w_j \) : score assigné à la feuille \( j \),
- \( \gamma \) : pénalité par feuille (favorise les arbres compacts),
- \( \lambda \) : régularisation L2 (réduit les poids extrêmes).

---

## 🔧 Approximation par expansion de Taylor

Pour optimiser efficacement, XGBoost applique un **développement de Taylor d'ordre 2** autour de \( \hat{y}_i^{(t-1)} \) :

$$
\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2 \right] + \Omega(f_t)
$$

avec :
- \( g_i = \frac{\partial l(y_i, \hat{y}_i)}{\partial \hat{y}_i} \) (gradient),
- \( h_i = \frac{\partial^2 l(y_i, \hat{y}_i)}{\partial \hat{y}_i^2} \) (hessienne).

Cette approximation permet de construire les arbres **en utilisant uniquement \( g_i \) et \( h_i \)**.

---

## 🌳 Construction de l’arbre et gain de split

Pour chaque split candidat, XGBoost calcule le **gain** suivant :

$$
\text{Gain} = \frac{1}{2} \left[
\frac{G_L^2}{H_L + \lambda} +
\frac{G_R^2}{H_R + \lambda} -
\frac{(G_L + G_R)^2}{H_L + H_R + \lambda}
\right] - \gamma
$$

avec :
- \( G_L = \sum g_i \) dans la feuille gauche,
- \( H_L = \sum h_i \) dans la feuille gauche,
- \( G_R = \sum g_i \) dans la feuille droite,
- \( H_R = \sum h_i \) dans la feuille droite.

Le split est accepté si le **gain est positif** (c’est-à-dire supérieur à \( \gamma \)).

---

## 🍃 Valeur optimale des feuilles

Une fois l’arbre construit, la **valeur optimale d’une feuille** \( j \) est :

$$
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

où \( I_j \) est l’ensemble des exemples dans la feuille \( j \).

---

## 🔁 Mise à jour des prédictions

À chaque étape, les prédictions sont mises à jour par :

$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(x_i)
$$

où \( \eta \) est le **taux d’apprentissage** (learning rate).

---

## 🧠 Remarques

- XGBoost peut être utilisé pour la **régression** (MSE) ou la **classification binaire** (log loss).
- L’approche par second ordre permet une **optimisation rapide et précise**.
- La régularisation \( \gamma \) et \( \lambda \) joue un rôle crucial pour **contrôler la complexité** du modèle.

---

## 📎 Références

- Chen & Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*  
  https://arxiv.org/abs/1603.02754  
- Documentation officielle : https://xgboost.readthedocs.io/
