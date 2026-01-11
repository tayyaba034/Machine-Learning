# Phase 1: Machine Learning Foundations

This document covers the **core foundations of Machine Learning**. It is written in an **intuitive, beginner-friendly way**, while still being **technically correct**. This phase is critical because *every ML algorithm is built on these ideas*. Additionally, we include **small code snippets in Python** to show the practical logic behind the theory.

---

## 1. What is Machine Learning?

### Simple intuition
Machine Learning is about **teaching computers to learn from data**, instead of writing fixed rules.

**Traditional programming:**
```
Rules + Data → Output
```

**Machine Learning:**
```
Data + Output → Rules (Model)
```

The model discovers patterns by itself.

### Formal definition
> Machine Learning is a subset of Artificial Intelligence that enables systems to learn patterns from data and make predictions or decisions without being explicitly programmed.

### Code Example: Checking data
```python
# Python snippet to look at dataset
import pandas as pd

data = pd.read_csv('data.csv')
print(data.head())  # Shows first 5 rows
```
This shows your raw data before applying any ML algorithm.

---

## 2. Why Do We Need Machine Learning?

Rule-based systems fail when:
- Rules become too complex
- Data is large
- Patterns change over time

Examples:
- Spam detection
- Recommendation systems
- Fraud detection
- Speech and image recognition

ML adapts automatically as data grows.

---

## 3. Types of Machine Learning

### 3.1 Supervised Learning
**Idea:** Learn from labeled data

Example:
```python
# Sample supervised learning data
X = [[0], [1], [2], [3]]  # Features
y = [0, 0, 1, 1]          # Labels
```
Common tasks:
- Classification
- Regression

---

### 3.2 Unsupervised Learning
**Idea:** No labels, only raw data

Example:
```python
from sklearn.cluster import KMeans
X = [[1], [2], [3], [10], [11]]
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)  # Cluster assignments
```

---

### 3.3 Semi-Supervised Learning
**Idea:** Small labeled dataset + large unlabeled dataset

---

### 3.4 Reinforcement Learning
**Idea:** Learn by trial and error using rewards

Pseudo-code:
```python
# Agent receives state, takes action, receives reward
# Q-learning / Policy gradient updates the model
state = env.reset()
for step in range(100):
    action = agent.choose_action(state)
    next_state, reward = env.step(action)
    agent.update(state, action, reward, next_state)
    state = next_state
```

---

## 4. ML Workflow

1. Problem definition
2. Data collection
3. Data preprocessing
4. Feature engineering
5. Model selection
6. Training
7. Evaluation
8. Deployment / inference

**Python snippet:**
```python
# Example skeleton workflow
# 1. Load Data
import pandas as pd
X = pd.read_csv('features.csv')
y = pd.read_csv('labels.csv')

# 2. Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 5. Data in Machine Learning

### Types of data
- Numerical (age, salary)
- Categorical (gender, city)
- Text
- Images

### Common data issues
- Missing values
- Outliers
- Noise
- Imbalanced data

**Python snippet:**
```python
# Handling missing values
X.fillna(X.mean(), inplace=True)
```

---

## 6. Training, Validation, and Test Sets

Typical split:
```python
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
```

---

## 7. Overfitting and Underfitting

**Intuition:**
- Underfitting → model too simple
- Overfitting → model too complex

**Python demonstration:**
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=1)  # Underfitting example
model.fit(X_train, y_train)
```
---

## 8. Bias–Variance Tradeoff

- High bias → underfitting
- High variance → overfitting

Visualize in Python:
```python
import matplotlib.pyplot as plt
# Plot learning curve or error vs complexity
```

---

## 9. Model Evaluation Basics

**Regression:** MSE, MAE, R²
```python
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(mse)
```

**Classification:** Accuracy, Precision, Recall
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
```

---

## 10. Key Takeaways of Phase 1

- ML is data-driven learning
- Data quality > model choice
- Overfitting understanding is critical
- Workflow knowledge is fundamental

---

## What Comes Next?

**Phase 2:** Linear Regression deep dive
- Theory + math intuition
- Python implementation with examples
- Understanding cost function and gradient descent

---

**End of Phase 1**

