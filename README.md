# 🧠 Breast Cancer Classification using SVM (Support Vector Machine)

This project demonstrates the use of **Support Vector Machines (SVM)** for binary classification using the **Breast Cancer Wisconsin dataset**. The project includes training SVMs with both **linear** and **non-linear (RBF)** kernels, visualizing decision boundaries, tuning hyperparameters, and evaluating performance with cross-validation.

---

## 📊 Dataset

- **Dataset**: Breast Cancer Wisconsin Diagnostic Dataset
- **Source**: `sklearn.datasets.load_breast_cancer`
- **Target Classes**:
  - `0` – Malignant (cancerous)
  - `1` – Benign (non-cancerous)
- **Features**: 30 numerical predictors (e.g., radius, texture, area)

---

## 🧰 Tools & Libraries

- Python
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn

---

## ✅ Project Objectives

### 1. Load and Prepare Data
- Loaded the built-in breast cancer dataset
- Scaled features using `StandardScaler`
- Split into train and test sets

### 2. Train SVM Classifiers
- **Linear SVM**: Kernel = `'linear'`, C = 1
- **Non-linear SVM**: Kernel = `'rbf'`, gamma = `'scale'`

### 3. Visualize Decision Boundary
- Applied PCA to reduce data to 2D
- Plotted decision boundary using meshgrid and contour plot

### 4. Hyperparameter Tuning
- Used `GridSearchCV` to optimize:
  - `C`: Regularization (0.1, 1, 10)
  - `gamma`: Kernel coefficient (`scale`, 0.01, 0.001)
  - `kernel`: Only `'rbf'`

### 5. Evaluate Using Cross-Validation
- Used 5-fold cross-validation on best model
- Reported average accuracy

---

## 📈 Performance

| Model      | Accuracy (Test) | CV Accuracy (mean) |
|------------|------------------|---------------------|
| SVM Linear | ~96%             | —                   |
| SVM RBF    | ~97–98%          | ~98.2%              |

> Note: Results may slightly vary based on random seed.

---

## 📂 File Structure

```plaintext
.
├── svm_breast_cancer.py         # Full project code
├── README.md                    # This file
