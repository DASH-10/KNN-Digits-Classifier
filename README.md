# KNN Digit Classifier

**Course:** Artificial Intelligence / Machine Learning  
**Topic:** Image Classification with K-Nearest Neighbors (KNN)  
**Student Name:** OMAR A.M. ISSA  
**Student No:** 220212901  
**Instructor:** Dr. Ramin Abbaszadi

---

## Project Description
This project implements a K-Nearest Neighbors (KNN) classifier from scratch and uses it to recognize handwritten digits from scikit-learn's `load_digits()` dataset (8x8 images). It evaluates different k values and distance metrics (L1 and L2), then compares the custom KNN results with scikit-learn's KNN for validation.

## Purpose
- Practice implementing the KNN algorithm without relying on library classifiers.
- Analyze how k and distance metrics affect classification accuracy.
- Compare a custom implementation against scikit-learn's reference model.

## What It Does
- Loads and normalizes the `load_digits()` dataset.
- Splits data into training and test sets.
- Runs KNN classification with L1 (Manhattan) and L2 (Euclidean) distances.
- Evaluates accuracy across multiple k values.
- Generates plots and tables that summarize performance.

## Code Structure
```
KNN_Project/
  knn_classifier.py        # Custom KNN implementation
  experiments.py           # Runs experiments and saves results
  visualization.py         # Plotting helpers
  main.ipynb               # Notebook version / exploration
  results/                 # Generated plots and tables
  README.md
```

## Requirements
- Python 3.8+
- numpy
- scikit-learn
- matplotlib
- seaborn (optional, for nicer heatmaps)

Install dependencies:
```bash
pip install numpy scikit-learn matplotlib seaborn
```

## How to Run
From the `KNN_Project` directory:
```bash
python experiments.py
```
This will generate output files in `results/`.

Optional (notebook):
```bash
jupyter notebook main.ipynb
```

## Outputs
Generated in `KNN_Project/results/`:
- `confusion_matrix.png`
- `sample_predictions.png`
- `k_value_analysis.png`
- `distance_comparison.png`
- `comparison_table.png`
