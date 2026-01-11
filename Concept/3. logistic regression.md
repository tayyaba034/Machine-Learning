# Logistic Regression (Supervised Learning - Classification)

This document covers **Logistic Regression**, a fundamental supervised learning algorithm used for **binary classification**
---

## 1. What is Logistic Regression?

**Intuition:**
- Logistic Regression predicts the **probability that an input belongs to a certain class**.
- Example: Predict if an email is **spam (1)** or **not spam (0)**.
- Unlike Linear Regression, the output is **between 0 and 1**.

**Equation:**
```
z = w0 + w1*x1 + w2*x2 + ... + wn*xn
ŷ = σ(z) = 1 / (1 + e^(-z))
```
Where:
- z = linear combination of inputs
- σ(z) = sigmoid function maps output to probability
- ŷ = predicted probability

---

## 2. Sigmoid Function

- Converts any real number to a value between 0 and 1.
- Output > 0.5 → class 1
- Output ≤ 0.5 → class 0

**Python Example:**
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
y = sigmoid(z)
plt.plot(z, y)
plt.title('Sigmoid Function')
plt.show()
```
**Explanation:**
- Smooth S-shaped curve
- Helps interpret model output as probability

---

## 3. Cost Function (Cross-Entropy / Log Loss)

- Measures how well predicted probabilities match actual labels

```
J(w) = -(1/n) * Σ [y*log(ŷ) + (1-y)*log(1-ŷ)]
```
- Minimizing this cost gives **best weights**
- MSE is not ideal for classification

**Python Example (sklearn computes automatically):**
```python
from sklearn.metrics import log_loss
y_true = [0,1,1,0]
y_pred = [0.1,0.9,0.8,0.2]
loss = log_loss(y_true, y_pred)
print('Log Loss:', loss)
```

---

## 4. Training / Gradient Descent

- Initialize weights randomly
- Update weights using **gradient of cost function**
- Repeat until cost converges

**Intuition:**
- Moves weights in direction to **maximize likelihood of correct classification**

---

## 5. Python Implementation (Binary Classification)

**Example: Predict if student passes based on hours studied**
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Feature: Hours studied
X = np.array([[1],[2],[3],[4],[5]])
# Label: Pass=1, Fail=0
y = np.array([0,0,0,1,1])

# Train model
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities
probs = model.predict_proba(X)[:,1]
print('Predicted probabilities:', probs)

# Predict classes
preds = model.predict(X)
print('Predicted classes:', preds)
```
**Explanation:**
- `predict_proba` gives probability of class 1
- `predict` gives the final class based on 0.5 threshold

---

## 6. Evaluation Metrics

**Binary Classification Metrics:**
- Accuracy = correct predictions / total
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-Score = harmonic mean of precision and recall

**Python Example:**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y, preds)
precision = precision_score(y, preds)
recall = recall_score(y, preds)
f1 = f1_score(y, preds)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1)
```

---

## 7. Key Takeaways

- Logistic Regression is for **classification**, outputs **probabilities**
- Sigmoid function ensures output between 0 and 1
- Cost function is **log loss / cross-entropy**
- Evaluation requires classification metrics, not MSE
- Very interpretable and widely used for binary problems

---

## Next Steps

After mastering Logistic Regression:
1. Multi-class Classification (Softmax Regression)
2. Polynomial / Non-linear features
3. Regularization (Ridge, Lasso, ElasticNet) to prevent overfitting

---


