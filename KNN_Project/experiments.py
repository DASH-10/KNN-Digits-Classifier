import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from knn_classifier import KNNClassifier
from visualization import (
    plot_confusion_matrix, plot_sample_predictions,
    plot_k_analysis, plot_distance_comparison, create_comparison_table
)
import os

os.makedirs('results', exist_ok=True)

def load_data():
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0  # normalize to 0-1
    y = digits.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test

def test_mnist_basic():
    print("=" * 50)
    print("Görev 1.2: MNIST Digits Testi")
    print("=" * 50)
    X_train, X_test, y_train, y_test = load_data()
    model = KNNClassifier(k=3, distance_metric='l2')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy (k=3, L2): {acc:.4f}")
    plot_confusion_matrix(y_test, y_pred, save_path='results/confusion_matrix.png')
    plot_sample_predictions(X_test, y_test, y_pred, n_samples=10, save_path='results/sample_predictions.png')
    return acc

def analyze_k_values():
    print("\n" + "=" * 50)
    print("Görev 1.3a: K Değeri Analizi")
    print("=" * 50)
    X_train, X_test, y_train, y_test = load_data()
    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    accuracies = []
    for k in k_values:
        model = KNNClassifier(k=k, distance_metric='l2')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"k={k:<2d} -> accuracy={acc:.4f}")
    plot_k_analysis(k_values, accuracies, save_path='results/k_value_analysis.png')
    return k_values, accuracies

def compare_distance_metrics():
    print("\n" + "=" * 50)
    print("Görev 1.3b: Mesafe Metriği Karşılaştırması")
    print("=" * 50)
    X_train, X_test, y_train, y_test = load_data()
    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    l1_accs, l2_accs = [], []
    for k in k_values:
        m1 = KNNClassifier(k=k, distance_metric='l1')
        m1.fit(X_train, y_train)
        acc1 = accuracy_score(y_test, m1.predict(X_test))
        m2 = KNNClassifier(k=k, distance_metric='l2')
        m2.fit(X_train, y_train)
        acc2 = accuracy_score(y_test, m2.predict(X_test))
        l1_accs.append(acc1)
        l2_accs.append(acc2)
        print(f"k={k:<2d} -> L1={acc1:.4f} | L2={acc2:.4f} | Diff={acc2-acc1:.4f}")
    plot_distance_comparison(k_values, l1_accs, l2_accs, save_path='results/distance_comparison.png')
    create_comparison_table(k_values, l1_accs, l2_accs, save_path='results/comparison_table.png')
    return k_values, l1_accs, l2_accs

def compare_with_sklearn():
    print("\n" + "=" * 50)
    print("Bölüm 2: Sklearn Karşılaştırması")
    print("=" * 50)
    from sklearn.neighbors import KNeighborsClassifier as SKKNN
    X_train, X_test, y_train, y_test = load_data()
    t0 = time.time()
    my_knn = KNNClassifier(k=3, distance_metric='l2')
    my_knn.fit(X_train, y_train)
    my_pred = my_knn.predict(X_test)
    my_time = time.time() - t0
    my_acc = accuracy_score(y_test, my_pred)
    t1 = time.time()
    sk_knn = SKKNN(n_neighbors=3, metric='minkowski', p=2, weights='uniform')
    sk_knn.fit(X_train, y_train)
    sk_pred = sk_knn.predict(X_test)
    sk_time = time.time() - t1
    sk_acc = accuracy_score(y_test, sk_pred)
    print(f"MyKNN  -> acc={my_acc:.4f} | time={my_time*1000:.2f} ms")
    print(f"SkKNN  -> acc={sk_acc:.4f} | time={sk_time*1000:.2f} ms")
    return (my_acc, my_time), (sk_acc, sk_time)

if __name__ == "__main__":
    acc = test_mnist_basic()
    k_values, k_accs = analyze_k_values()
    k_values2, l1_accs, l2_accs = compare_distance_metrics()
    my_stats, sk_stats = compare_with_sklearn()
