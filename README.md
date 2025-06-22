# Machine Learning from Scratch — Boosting Didactique

> Auteur : **Manda Rostand**  
> Profession : Data Scientist (5 ans d'expérience)  
> Contact : rostandsurel@yahoo.com

---

Ce dépôt GitHub a pour vocation d'être un référentiel **évolutif et pédagogique** dédié à l’apprentissage et à l’implémentation des principaux algorithmes de boosting. Il s’adresse aux étudiants, professionnels ou passionnés de machine learning qui souhaitent **comprendre les fondements mathématiques et algorithmiques** de ces modèles en les **reproduisant from scratch**.

---

## 🚀 Objectif

L’objectif principal est de **favoriser la compréhension approfondie** de modèles de boosting en partant de zéro :
- en décortiquant leurs mécanismes internes,
- en expliquant les intuitions mathématiques et algorithmiques,
- et en les implémentant sans dépendre de bibliothèques toutes faites comme scikit-learn ou XGBoost directement.

---

## 📚 Contenu du dépôt

### 🔹 [XGBoost](./xgboost_from_scratch)
- Cours complet : boosting, gradient, loss functions, régularisation
- Dérivation mathématique de l’algorithme
- Implémentation étape par étape en Python
- Visualisation des arbres
- Cas d’usage simple (classification ou régression)

### 🔹 [CatBoost](./catboost_from_scratch)
- Cours sur le traitement des variables catégorielles, encodage ordonné
- Spécificités de l'algorithme : target statistics, permutations
- Implémentation complète from scratch
- Comparaison avec XGBoost

### 🔹 [LightGBM](./lightgbm_from_scratch)
- Théorie sur histogram-based decision trees et leaf-wise growth
- Optimisations mémoire et temps
- Implémentation avec gestion des bins
- Benchmark basique

---

## 🧠 Méthodologie

Chaque sous-répertoire contient :
- Un dossier `cours/` contenant des fichiers `.md` avec :
  - Définitions formelles
  - Formules clés (dérivées, gradients, hessians, etc.)
  - Explications étape par étape
- Un dossier `implementation/` contenant :
  - Le code Python `from scratch`
  - Des blocs bien commentés
  - Des tests sur des jeux de données jouets
- Des illustrations et des schémas seront ajoutés pour mieux visualiser les concepts.

---

## 📌 Pourquoi ce dépôt ?

La majorité des projets ML reposent sur des bibliothèques haut niveau. Ce dépôt répond à un besoin différent :
- **Démythifier** les modèles complexes
- **Développer une intuition forte**
- **Se préparer à des entretiens techniques ML/DS**
- **Mieux comprendre les hyperparamètres et les choix d’implémentation**

---

## 🛠️ À venir

- Ajout de notebooks Jupyter pour tester les implémentations
- Visualisation dynamique des arbres
- Comparaison des performances vs bibliothèques standards
- Extensions : AdaBoost, GradientBoosting classique, etc.

---

## ⚖️ Licence

Ce dépôt est sous licence MIT. Vous êtes libres de l'utiliser, le modifier ou le redistribuer avec mention de l'auteur.

Voir [LICENSE](./LICENSE) pour plus d'informations.

---

## 🙌 Remerciements

Merci à la communauté open source pour l’inspiration continue. Ce projet est également un outil d’auto-formation et de transmission.

---

## 📩 Contact

Si vous avez des questions, suggestions ou souhaitez collaborer :

- 📧 Email : rostandsurel@yahoo.com
- 💼 LinkedIn : [https://www.linkedin.com/in/rostand-surel/](https://www.linkedin.com/in/rostand-surel/)

---
