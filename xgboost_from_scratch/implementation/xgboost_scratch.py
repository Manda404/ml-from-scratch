"""
XGBoost From Scratch - Implémentation pédagogique

Auteur : Manda Rostand
Date : Juin 2025
Description : Implémentation simplifiée de l'algorithme XGBoost pour la régression.

---

🧱 Étapes de construction :
1. Définir une classe XGBoostRegressor avec les paramètres principaux
2. Implémenter la loss MSE avec calcul des gradients et hessiennes
3. Construire un arbre de décision récursif basé sur le gain de XGBoost
4. Ajouter régularisation L2 (lambda) et pénalisation de la complexité (gamma)
5. Implémenter la prédiction individuelle pour un arbre
6. Combiner plusieurs arbres en boosting via mise à jour incrémentale
"""

import numpy as np

class XGBoostRegressor:
    """
    Implémentation from scratch de l’algorithme XGBoost pour régression.
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=10, lambda_=1.0, gamma=0.0):
        """
        Initialise le modèle avec ses hyperparamètres.

        Parameters:
        - n_estimators : int, nombre d’arbres à booster
        - learning_rate : float, taux d’apprentissage
        - max_depth : int, profondeur maximale de chaque arbre
        - min_samples_split : int, minimum d’échantillons pour split
        - lambda_ : float, régularisation L2
        - gamma : float, pénalisation pour contrôler la croissance des arbres
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
        Entraîne le modèle sur les données X et y.

        Parameters:
        - X : ndarray, shape (n_samples, n_features)
        - y : ndarray, shape (n_samples,)
        """
        self.base_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.base_prediction)

        for _ in range(self.n_estimators):
            gradient = self._gradient(y, y_pred) # calcul du gradient
            hessian = self._hessian(y, y_pred)
            tree = self._build_tree(X, gradient, hessian, depth=0)
            self.trees.append(tree)
            y_pred += self.learning_rate * self._predict_tree(X, tree)

    def predict(self, X):
        """
        Prédit les valeurs cibles pour les entrées X.

        Parameters:
        - X : ndarray, shape (n_samples, n_features)
        Returns:
        - y_pred : ndarray, shape (n_samples,)
        """
        y_pred = np.full((X.shape[0],), self.base_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * self._predict_tree(X, tree)
        return y_pred

    def _gradient(self, y, y_pred):
        """
        Calcule les gradients pour la loss MSE.

        Returns:
        - g : ndarray
        """
        return y_pred - y

    def _hessian(self, y, y_pred):
        """
        Calcule les hessiennes (constantes pour MSE).

        Returns:
        - h : ndarray
        """
        return np.ones_like(y)

    def _build_tree(self, X, g, h, depth):
        """
        Construit récursivement un arbre de décision basé sur le gain de XGBoost.

        Returns:
        - tree : dict représentant un nœud ou une feuille
        """
        n_samples, n_features = X.shape

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = -np.sum(g) / (np.sum(h) + self.lambda_)
            return {"type": "leaf", "value": leaf_value}

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
            leaf_value = -np.sum(g) / (np.sum(h) + self.lambda_)
            return {"type": "leaf", "value": leaf_value}

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
        Calcule le gain d’un split selon la formule XGBoost.

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
        Applique un arbre de décision à chaque ligne de X.

        Returns:
        - preds : ndarray, shape (n_samples,)
        """
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            preds[i] = self._traverse_tree(X[i], tree)
        return preds

    def _traverse_tree(self, x, tree):
        """
        Parcourt un arbre pour prédire une seule instance x.

        Returns:
        - valeur de la feuille
        """
        if tree["type"] == "leaf":
            return tree["value"]

        if x[tree["feature"]] <= tree["threshold"]:
            return self._traverse_tree(x, tree["left"])
        else:
            return self._traverse_tree(x, tree["right"])
