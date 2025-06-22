"""
XGBoost From Scratch - Classification Binaire

Auteur : Manda Rostand  
Date : Juin 2025  
Description : Implémentation simplifiée de l'algorithme XGBoost pour la classification binaire.

---

🧱 Étapes de construction :
1. Définir une classe XGBoostClassifier avec les hyperparamètres
2. Implémenter la loss logistique (log loss) avec gradients et hessiennes
3. Construire récursivement un arbre basé sur le gain de XGBoost
4. Utiliser la log-odds comme prédiction intermédiaire
5. Convertir les log-odds en probabilité via la sigmoïde
6. Prédire les classes en comparant la probabilité à un seuil
"""

import numpy as np

class XGBoostClassifier:
    """
    Implémentation from scratch de XGBoost pour classification binaire (log loss).
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=10, lambda_=1.0, gamma=0.0):
        """
        Initialise le classifieur avec les hyperparamètres de boosting.

        Parameters:
        - n_estimators : int, nombre total d’arbres à entraîner
        - learning_rate : float, taux d’apprentissage
        - max_depth : int, profondeur maximale des arbres
        - min_samples_split : int, seuil de split minimal
        - lambda_ : float, régularisation L2
        - gamma : float, gain minimum requis pour splitter un nœud
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_ = lambda_
        self.gamma = gamma
        self.trees = []

    def fit(self, X, y):
        """
        Entraîne le modèle sur les données d'entraînement (X, y).

        Parameters:
        - X : ndarray, shape (n_samples, n_features)
        - y : ndarray, shape (n_samples,), valeurs 0 ou 1
        """
        self.base_prediction = 0.0  # log(odds) initial = 0 → proba = 0.5
        y_pred_log_odds = np.full(y.shape, self.base_prediction)

        for _ in range(self.n_estimators):
            p = self._sigmoid(y_pred_log_odds)
            gradient = p - y                      # dérivée première
            hessian = p * (1 - p)                 # dérivée seconde

            tree = self._build_tree(X, gradient, hessian, depth=0)
            self.trees.append(tree)

            update = self._predict_tree(X, tree)
            y_pred_log_odds -= self.learning_rate * update

    def predict_proba(self, X):
        """
        Retourne la probabilité (sigmoïde des log-odds).

        Parameters:
        - X : ndarray
        Returns:
        - probs : ndarray, probabilité pour la classe positive
        """
        y_pred = np.full(X.shape[0], self.base_prediction)
        for tree in self.trees:
            y_pred -= self.learning_rate * self._predict_tree(X, tree)
        return self._sigmoid(y_pred)

    def predict(self, X):
        """
        Prédit la classe (0 ou 1) en appliquant un seuil de 0.5.

        Parameters:
        - X : ndarray
        Returns:
        - classes : ndarray, valeurs binaires
        """
        return (self.predict_proba(X) >= 0.5).astype(int)

    def _sigmoid(self, x):
        """
        Fonction d'activation logistique.

        Parameters:
        - x : ndarray ou float
        Returns:
        - sigmoid(x)
        """
        return 1 / (1 + np.exp(-x))

    def _build_tree(self, X, g, h, depth):
        """
        Construction récursive de l’arbre basé sur le gain.

        Parameters:
        - X : ndarray
        - g : gradients
        - h : hessiennes
        - depth : profondeur actuelle

        Returns:
        - dict représentant un arbre (noeud ou feuille)
        """
        n_samples, n_features = X.shape

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            value = np.sum(g) / (np.sum(h) + self.lambda_)
            return {"type": "leaf", "value": value}

        best_gain = -np.inf
        best_split = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                if len(g[left_idx]) == 0 or len(g[right_idx]) == 0:
                    continue

                gain = self._calculate_gain(g, h, left_idx, right_idx)
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature": feature,
                        "threshold": threshold,
                        "left_idx": left_idx,
                        "right_idx": right_idx
                    }

        if best_gain < self.gamma or best_split is None:
            value = np.sum(g) / (np.sum(h) + self.lambda_)
            return {"type": "leaf", "value": value}

        left = self._build_tree(X[best_split["left_idx"]], g[best_split["left_idx"]], h[best_split["left_idx"]], depth + 1)
        right = self._build_tree(X[best_split["right_idx"]], g[best_split["right_idx"]], h[best_split["right_idx"]], depth + 1)

        return {
            "type": "node",
            "feature": best_split["feature"],
            "threshold": best_split["threshold"],
            "left": left,
            "right": right
        }

    def _calculate_gain(self, g, h, left_idx, right_idx):
        """
        Calcule le gain de split basé sur la formulation de XGBoost.

        Parameters:
        - g : gradients
        - h : hessiennes
        - left_idx, right_idx : masques booléens

        Returns:
        - gain : float
        """
        G_L = np.sum(g[left_idx])
        H_L = np.sum(h[left_idx])
        G_R = np.sum(g[right_idx])
        H_R = np.sum(h[right_idx])
        G = G_L + G_R
        H = H_L + H_R

        gain = 0.5 * (
            (G_L ** 2) / (H_L + self.lambda_) +
            (G_R ** 2) / (H_R + self.lambda_) -
            (G ** 2) / (H + self.lambda_)
        ) - self.gamma

        return gain

    def _predict_tree(self, X, tree):
        """
        Applique un arbre à un batch de données X.

        Returns:
        - ndarray des prédictions
        """
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            preds[i] = self._traverse_tree(X[i], tree)
        return preds

    def _traverse_tree(self, x, tree):
        """
        Traverse un arbre pour une seule instance.

        Returns:
        - valeur prédite (log-odds update)
        """
        if tree["type"] == "leaf":
            return tree["value"]

        if x[tree["feature"]] <= tree["threshold"]:
            return self._traverse_tree(x, tree["left"])
        else:
            return self._traverse_tree(x, tree["right"])
