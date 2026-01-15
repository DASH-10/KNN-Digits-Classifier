import numpy as np
from collections import Counter


class KNNClassifier:
    """
    K-En Yakın Komşuluk (K-Nearest Neighbors) Sınıflandırıcı
    """

    def __init__(self, k=3, distance_metric='l2'):
        """
        KNN sınıflandırıcıyı başlatır

        Parameters:
        -----------
        k : int
            Komşu sayısı
        distance_metric : str
            Mesafe metriği ('l1' veya 'l2')
        """
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = int(k)
        self.distance_metric = distance_metric.lower()
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Training verisini kaydeder

        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Training özellikleri
        y : numpy array, shape (n_samples,)
            Training etiketler
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_features).")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D array with length equal to X.shape[0].")
        self.X_train = X
        self.y_train = y

    def compute_distances(self, X):
        """
        Test örnekleri ile training örnekleri arasındaki mesafeleri hesaplar

        Parameters:
        -----------
        X : numpy array, shape (n_test, n_features)
            Test örnekleri

        Returns:
        --------
        distances : numpy array, shape (n_test, n_train)
            Mesafe matrisi
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model not fitted. Call fit(X, y) before computing distances.")

        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]:
            raise ValueError("X must be 2D with the same number of features as training data.")

        if self.distance_metric == 'l1':
            # L1 (Manhattan) distance
            diffs = np.abs(X[:, None, :] - self.X_train[None, :, :])
            distances = diffs.sum(axis=2)
            return distances
        elif self.distance_metric == 'l2':
            # L2 (Euclidean) distance using expansion:
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (n_test, 1)
            Y_sq = np.sum(self.X_train ** 2, axis=1, keepdims=True).T  # (1, n_train)
            XY = X @ self.X_train.T  # (n_test, n_train)
            d2 = X_sq + Y_sq - 2.0 * XY
            np.maximum(d2, 0.0, out=d2)  # numerical stability
            return np.sqrt(d2, dtype=X.dtype)
        else:
            raise ValueError(f"Bilinmeyen mesafe metriği: {self.distance_metric}")

    def predict(self, X):
        """
        Test örnekleri için tahmin yapar

        Parameters:
        -----------
        X : numpy array, shape (n_test, n_features)
            Test örnekleri

        Returns:
        --------
        predictions : numpy array, shape (n_test,)
            Tahmin edilen etiketler
        """
        distances = self.compute_distances(X)  # (n_test, n_train)
        n_test = distances.shape[0]
        preds = np.empty(n_test, dtype=self.y_train.dtype)

        # Top-k komşuları seç (hızlı argpartition)
        neighbor_idx = np.argpartition(distances, self.k - 1, axis=1)[:, :self.k]  # (n_test, k)

        # Bu top-k içini mesafeye göre sırala
        row_indices = np.arange(n_test)[:, None]
        topk_sorted = neighbor_idx[row_indices, np.argsort(distances[row_indices, neighbor_idx], axis=1)]

        for i in range(n_test):
            labels = self.y_train[topk_sorted[i]]
            counts = Counter(labels)
            max_count = max(counts.values())
            candidates = [lab for lab, c in counts.items() if c == max_count]
            preds[i] = np.min(candidates)  # tie-break: en küçük etiket
        return preds

    def score(self, X, y):
        """
        Model accuracy'sini hesaplar

        Parameters:
        -----------
        X : numpy array, shape (n_test, n_features)
            Test özellikleri
        y : numpy array, shape (n_test,)
            Gerçek etiketler

        Returns:
        --------
        accuracy : float
            Doğruluk skoru (0-1 arası)
        """
        y = np.asarray(y)
        y_pred = self.predict(X)
        return float((y_pred == y).mean())
