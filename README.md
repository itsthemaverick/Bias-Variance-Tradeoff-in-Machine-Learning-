# Bias–Variance Tradeoff in Machine Learning  
### Underfitting, Overfitting, Regularization & Cross-Validation (scikit-learn)

## 📌 Overview
This repository demonstrates one of the most fundamental challenges in Machine Learning — the **Bias–Variance Tradeoff** — using controlled experiments, visualizations, and production-style scikit-learn pipelines.

Instead of relying on definitions alone, this project builds models of increasing complexity, shows where and why they fail, and then applies **regularization** and **cross-validation** to systematically fix those failures.

The focus is on **generalization**, not just training accuracy.

---

## 🎯 Objectives
- Demonstrate underfitting and overfitting visually
- Explain bias and variance through loss behavior
- Apply L1 and L2 regularization to control complexity
- Use K-Fold cross-validation for model selection
- Implement everything using clean, reusable scikit-learn pipelines

---

## 🧠 Core Idea
Every supervised learning model minimizes a loss function on training data.  
However, minimizing training loss alone does not guarantee good performance on unseen data.

This project explores how:
- **Simple models** suffer from high bias (underfitting)
- **Complex models** suffer from high variance (overfitting)
- **Regularization** constrains model complexity
- **Cross-validation** estimates true generalization error

---

## 🗂️ Project Structure

bias-variance-demo/
│
├── data/
│ └── generate_data.py
│
├── models/
│ ├── underfit.py
│ ├── overfit.py
│ ├── regularization.py
│ └── cross_validation.py
│
├── utils/
│ └── plotting.py
│
├── requirements.txt
└── README.md

---

## 📊 Dataset
A synthetic non-linear dataset is generated using a sine function with Gaussian noise:

y = sin(x) + ε

Why synthetic data?
- Eliminates ambiguity from real-world noise
- Makes bias–variance behavior visually clear
- Enables controlled experimentation

---

## 🔴 Underfitting (High Bias)
**Model:** Polynomial Regression (degree = 1)

**Behavior:**
- Model is too simple to capture non-linear structure
- High training error
- High validation error

**Insight:**  
Strong assumptions lead to systematic errors.

---

## 🔵 Overfitting (High Variance)
**Model:** Polynomial Regression (degree = 15)

**Behavior:**
- Model fits noise in the training data
- Very low training error
- Poor generalization to unseen data

**Insight:**  
Memorization replaces learning when model complexity is too high.

---

## 🧩 Regularization (Controlling Model Complexity)
To reduce overfitting, a penalty term is added to the loss function:

Loss = Data Loss + λ × Regularization Term

### Ridge Regression (L2)
- Penalizes squared magnitude of weights
- Produces smooth, stable models
- Retains all features

### Lasso Regression (L1)
- Penalizes absolute magnitude of weights
- Forces some weights to zero
- Performs implicit feature selection

**Engineering Trade-off:**
- Ridge → stability
- Lasso → sparsity

---

## 🔁 Cross-Validation (Model Selection)
K-Fold Cross-Validation is used to estimate generalization error by:
- Training on K−1 folds
- Validating on the remaining fold
- Repeating this process K times

The resulting error curve clearly shows:
- High error at low complexity (underfitting)
- Minimum error at optimal complexity
- Rapid error increase at high complexity (overfitting)

This is how model complexity is selected in real-world ML systems.

---

## 🛠️ Tools & Technologies
- Python
- NumPy
- Matplotlib
- scikit-learn
- Pipeline-based modeling
- K-Fold Cross-Validation

---

## 📚 Key Takeaways
- Training accuracy alone is misleading
- Generalization error matters more than training loss
- Model complexity must be controlled
- Regularization reduces variance in a principled way
- Cross-validation is essential for reliable evaluation

---

## 🔮 Possible Extensions
- Learning curves (training vs validation loss)
- Logistic regression and classification version
- Hyperparameter tuning using GridSearchCV
- Bias–variance analysis in neural networks

---

## 👤 Author : ItsTheMaverick 
Built to demonstrate **engineering-level understanding of Machine Learning fundamentals**, with emphasis on clarity, rigor, and real-world modeling practices.

⭐ If you found this project useful, consider starring the repository.


