import numpy as np

# =============================
# 🎯 Objectif : Implémentation d’un Mini-XGBoost (un seul arbre, sans régularisation avancée)
# 
# Structure de Mini-XGBoost :
# - compute_gradients : calcule les gradients et hessiennes (MSE)
# - calculate_gain : calcule le gain de split basé sur les gradients
# - find_best_split : identifie la meilleure coupure sur les features
# - TreeNode : structure d’arbre binaire pour les prédictions
# - build_tree : construit l’arbre récursivement avec logique de gain
# - predict_tree / predict : prédictions récursives avec l’arbre
# - train_boosting_tree : entraîne un arbre sur les résidus actuels
# 
# Diagramme d'interaction des fonctions :
# 
#        train_boosting_tree
#                │
#                ▼
#       compute_gradients
#                │
#                ▼
#           build_tree
#                │
#                ├──> find_best_split
#                │         └──> calculate_gain
#                │
#                └──> récursion build_tree
# 
#        predict
#           │
#           └──> predict_tree (appel récursif)
# =============================

# =============================
# 🧮 Étape 1 : Fonctions de perte et gradients
# =============================
def compute_gradients(y_true, y_pred):
    """
    Calcule les gradients et hessiennes de la fonction de perte quadratique (MSE).

    Loss(y, y_pred) = 1/2 * (y - y_pred)^2
    Gradient = y_pred - y
    Hessienne = 1 (car dérivée seconde de la loss quadratique est constante)

    Args:
        y_true (np.array): Les vraies valeurs cibles.
        y_pred (np.array): Les prédictions courantes du modèle.

    Returns:
        gradients (np.array): Les gradients pour chaque échantillon.
        hessians (np.array): Les hessiennes pour chaque échantillon (ici, vecteur de 1).
    """
    gradients = y_pred - y_true
    hessians = np.ones_like(y_true)
    return gradients, hessians

# =============================
# 🌳 Étape 2 : Calcul du gain pour un split donné
# =============================
def calculate_gain(G_left, H_left, G_right, H_right, lambda_):
    """
    Calcule le gain obtenu en réalisant un split selon la formule XGBoost (sans régularisation L1).

    Gain = 1/2 * [(G_L^2)/(H_L + lambda) + (G_R^2)/(H_R + lambda) - (G^2)/(H + lambda)]

    Args:
        G_left (float): Somme des gradients à gauche.
        H_left (float): Somme des hessiennes à gauche.
        G_right (float): Somme des gradients à droite.
        H_right (float): Somme des hessiennes à droite.
        lambda_ (float): Paramètre de régularisation L2.

    Returns:
        gain (float): Gain du split considéré.
    """
    def score(G, H):
        return (G ** 2) / (H + lambda_)

    gain = 0.5 * (score(G_left, H_left) + score(G_right, H_right) -
                  score(G_left + G_right, H_left + H_right))
    return gain

# =============================
# 🔍 Étape 3 : Recherche du meilleur split
# =============================
def find_best_split(X, gradients, hessians, lambda_, min_samples_leaf=1):
    """
    Recherche le meilleur split en parcourant toutes les colonnes/features et seuils uniques.

    Args:
        X (np.array): Données d'entrée.
        gradients (np.array): Gradients des échantillons.
        hessians (np.array): Hessiennes des échantillons.
        lambda_ (float): Paramètre de régularisation.
        min_samples_leaf (int): Nombre minimum d'échantillons par feuille.

    Returns:
        best_feature (int): Index de la feature pour le meilleur split.
        best_threshold (float): Valeur seuil pour le split.
        best_gain (float): Gain correspondant au meilleur split.
    """
    best_gain = -np.inf
    best_feature = None
    best_threshold = None

    for feature_idx in range(X.shape[1]):
        feature_values = X[:, feature_idx]
        thresholds = np.unique(feature_values)

        for threshold in thresholds:
            left_mask = feature_values <= threshold
            right_mask = ~left_mask

            if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
                continue

            G_left, H_left = gradients[left_mask].sum(), hessians[left_mask].sum()
            G_right, H_right = gradients[right_mask].sum(), hessians[right_mask].sum()

            gain = calculate_gain(G_left, H_left, G_right, H_right, lambda_)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_gain

# =============================
# 🌿 Étape 4 : Construction de l'arbre de décision
# =============================
class TreeNode:
    """
    Classe représentant un noeud dans un arbre de décision.
    """
    def __init__(self, is_leaf=False, value=None, feature=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

def build_tree(X, gradients, hessians, depth, max_depth, lambda_, gamma):
    """
    Construction récursive de l'arbre de décision via split selon les gradients/hessiennes.

    Args:
        X (np.array): Données d'entrée.
        gradients (np.array): Gradients à chaque point.
        hessians (np.array): Hessiennes à chaque point.
        depth (int): Profondeur actuelle.
        max_depth (int): Profondeur maximale de l'arbre.
        lambda_ (float): Paramètre de régularisation.
        gamma (float): Seuil minimal de gain pour un split.

    Returns:
        TreeNode: Noeud racine de l'arbre construit.
    """
    if depth >= max_depth:
        leaf_value = -gradients.sum() / (hessians.sum() + lambda_)
        return TreeNode(is_leaf=True, value=leaf_value)

    feature, threshold, gain = find_best_split(X, gradients, hessians, lambda_)
    if feature is None or gain < gamma:
        leaf_value = -gradients.sum() / (hessians.sum() + lambda_)
        return TreeNode(is_leaf=True, value=leaf_value)

    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask

    left = build_tree(X[left_mask], gradients[left_mask], hessians[left_mask],
                      depth + 1, max_depth, lambda_, gamma)
    right = build_tree(X[right_mask], gradients[right_mask], hessians[right_mask],
                       depth + 1, max_depth, lambda_, gamma)

    return TreeNode(is_leaf=False, feature=feature, threshold=threshold,
                    left=left, right=right)

# =============================
# 🔮 Étape 5 : Prédiction via l’arbre
# =============================
def predict_tree(x, node):
    """
    Prédit une valeur pour un seul échantillon en parcourant l’arbre récursivement.
    """
    if node.is_leaf:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_tree(x, node.left)
    else:
        return predict_tree(x, node.right)

def predict(X, tree):
    """
    Prédit pour tous les échantillons en appelant predict_tree.
    """
    return np.array([predict_tree(x, tree) for x in X])

# =============================
# 🚀 Étape 6 : Entraînement d’un arbre boosting
# =============================
def train_boosting_tree(X, y, y_pred, max_depth=3, lambda_=1.0, gamma=0.0):
    """
    Entraîne un arbre de boosting sur les résidus (en utilisant les gradients).

    Args:
        X (np.array): Données d'entrée.
        y (np.array): Valeurs cibles.
        y_pred (np.array): Prédictions actuelles (initialisées par la moyenne).
        max_depth (int): Profondeur maximale de l'arbre.
        lambda_ (float): Régularisation L2.
        gamma (float): Gain minimal pour split.

    Returns:
        TreeNode: Arbre entraîné.
    """
    gradients, hessians = compute_gradients(y, y_pred)
    tree = build_tree(X, gradients, hessians, depth=0, max_depth=max_depth, lambda_=lambda_, gamma=gamma)
    return tree

# =============================
# ✅ Exemple d'utilisation
# =============================
if __name__ == "__main__":
    # Jeu de données jouet
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 1.3, 3.75, 2.25])

    # Prédiction initiale = moyenne
    y_pred = np.full_like(y, y.mean(), dtype=float)

    # Entraînement d’un arbre
    tree = train_boosting_tree(X, y, y_pred, max_depth=2)

    # Nouvelle prédiction
    new_pred = predict(X, tree)

    # Mise à jour des prédictions
    y_pred += new_pred

    print("Nouvelles prédictions :", y_pred)
