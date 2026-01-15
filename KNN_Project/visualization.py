import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# seaborn varsa heatmap daha güzel görünür; yoksa matplotlib ile devam
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    if _HAS_SNS:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    else:
        plt.imshow(cm, cmap="Blues")
        for (i,j), val in np.ndenumerate(cm):
            plt.text(j, i, int(val), ha='center', va='center')
        plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_sample_predictions(X_test, y_test, y_pred, n_samples=10, save_path='results/sample_predictions.png'):
    n_samples = int(n_samples)
    n = X_test.shape[0]
    idx = np.random.choice(n, size=min(n_samples, n), replace=False)
    cols = min(5, n_samples)
    rows = int(np.ceil(len(idx) / cols))
    plt.figure(figsize=(3*cols, 3*rows))
    for k, i in enumerate(idx):
        img = X_test[i].reshape(8,8)
        t, p = y_test[i], y_pred[i]
        plt.subplot(rows, cols, k+1)
        plt.imshow(img, cmap='gray')
        color = "green" if t == p else "red"
        plt.title(f"True: {t} | Pred: {p}", color=color, fontsize=10)
        plt.axis('off')
    plt.suptitle("Sample Predictions", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_k_analysis(k_values, accuracies, save_path='results/k_value_analysis.png'):
    plt.figure(figsize=(6,4))
    plt.plot(k_values, accuracies, marker='o')
    best_i = int(np.argmax(accuracies))
    plt.scatter([k_values[best_i]], [accuracies[best_i]], s=80)
    plt.title("Accuracy vs K")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_distance_comparison(k_values, l1_accuracies, l2_accuracies, save_path='results/distance_comparison.png'):
    plt.figure(figsize=(6,4))
    plt.plot(k_values, l1_accuracies, marker='o', label='L1 (Manhattan)')
    plt.plot(k_values, l2_accuracies, marker='o', label='L2 (Euclidean)')
    plt.title("Distance Metric Comparison")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def create_comparison_table(k_values, l1_accuracies, l2_accuracies, save_path='results/comparison_table.png'):
    diffs = [round(l2 - l1, 4) for l1, l2 in zip(l1_accuracies, l2_accuracies)]
    col_labels = ["K Değeri", "L1 Accuracy", "L2 Accuracy", "Fark (L2-L1)"]
    table_data = [[k, round(l1,4), round(l2,4), d]
                  for k, l1, l2, d in zip(k_values, l1_accuracies, l2_accuracies, diffs)]
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.4*len(k_values)))
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    plt.title("Mesafe Metriği Karşılaştırma Tablosu", pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
