# --------------------------------------------
# SVM with Linear and RBF Kernel on Breast Cancer Dataset
# --------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------
# 1. Load and Prepare Dataset
# --------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("🔍 Data Shape:", X.shape)
print("Target classes:", data.target_names)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------
# 2. Train SVM (Linear and RBF Kernel)
# --------------------------------------------
svm_linear = SVC(kernel='linear', C=1)
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')

svm_linear.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)

print("\n🔎 Linear Kernel Classification Report:")
print(classification_report(y_test, svm_linear.predict(X_test)))

print("\n🔎 RBF Kernel Classification Report:")
print(classification_report(y_test, svm_rbf.predict(X_test)))

# --------------------------------------------
# 3. Visualize Decision Boundary (PCA to 2D)
# --------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

model_pca = SVC(kernel='rbf', C=1, gamma='scale')
model_pca.fit(X_train_pca, y_train_pca)

def plot_decision_boundary(X, y, model, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 5))
    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.tight_layout()
    plt.show()

plot_decision_boundary(X_test_pca, y_test_pca, model_pca, "SVM with RBF Kernel (PCA Reduced)")

# --------------------------------------------
# 4. Hyperparameter Tuning (C and gamma)
# --------------------------------------------
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("\n✅ Best Parameters from GridSearchCV:")
print(grid.best_params_)

# --------------------------------------------
# 5. Cross-Validation Performance
# --------------------------------------------
svm_best = grid.best_estimator_
cv_scores = cross_val_score(svm_best, X_scaled, y, cv=5)

print("\n📊 Cross-Validation Accuracy (Best Model):")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"All Fold Scores: {cv_scores}")
