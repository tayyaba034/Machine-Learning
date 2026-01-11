# Phase 4: K-Nearest Neighbors (KNN)

This document covers **K-Nearest Neighbors (KNN)**, a simple yet powerful algorithm in **supervised learning**. We will explain it **intuitively, mathematically, and graphically**, along with **Python implementation**.

---

## 1. What is KNN?

**Intuition:**
- KNN is a **distance-based classification/regression algorithm**.
- It predicts the label of a new data point based on the **labels of its K nearest neighbors** in the training set.

**Key Points:**
- Non-parametric: no assumptions about data distribution
- Lazy learning: no training phase, just stores the data
- Works for classification and regression

**Graphical Idea:**
```
+ Class A points
x Class B points
New point to classify
```
Imagine a 2D plot:
- Blue dots = Class A
- Red dots = Class B
- New point = green star
- Look at K nearest neighbors → majority vote = predicted class

---

## 2. Distance Metric

**Common metrics:**
- Euclidean Distance (most common)
```
d = sqrt((x1-x2)^2 + (y1-y2)^2)
```
- Manhattan Distance
- Minkowski Distance

**Python Example:**
```python
import numpy as np

point1 = np.array([1,2])
point2 = np.array([4,6])
distance = np.sqrt(np.sum((point1 - point2)**2))
print('Euclidean Distance:', distance)
```

---

## 3. KNN Algorithm (Step by Step)

1. Choose the number of neighbors, K
2. Calculate the distance between the new point and all training points
3. Sort the distances and select K nearest points
4. For classification: take **majority vote**
5. For regression: take **average of K neighbors**
6. Assign predicted value or class

**Graphical Representation:**
- Circle around new point
- Nearest neighbors inside circle
- Highlight majority class

---

## 4. Choosing K

- Small K → sensitive to noise → may overfit
- Large K → smoother decision boundary → may underfit

**Rule of Thumb:** K ≈ sqrt(n_samples)

---

## 5. Python Implementation (Classification)

**Example: Classify points into two classes based on 2D coordinates**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Sample data
X = np.array([[1,2],[2,3],[3,3],[6,5],[7,7],[8,6]])
y = np.array([0,0,0,1,1,1])  # 0=Class A, 1=Class B

# Train KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# New point to predict
new_point = np.array([[5,5]])
pred = model.predict(new_point)
print('Predicted Class:', pred[0])

# Visualization
plt.scatter(X[:3,0], X[:3,1], color='blue', label='Class A')
plt.scatter(X[3:,0], X[3:,1], color='red', label='Class B')
plt.scatter(new_point[0,0], new_point[0,1], color='green', marker='*', s=200, label='New Point')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('KNN Classification')
plt.show()
```
**Explanation:**
- Green star = new point
- Blue/Red = existing classes
- Algorithm calculates distance → looks at K nearest → predicts class

---

## 6. Advantages

- Simple to implement
- No assumptions about data
- Naturally handles multi-class problems

## 7. Disadvantages

- Slow for large datasets
- Sensitive to irrelevant features and scale
- Choosing K is sometimes tricky

---

## 8. Key Takeaways

- KNN is **intuitive and visual**
- Works on majority vote (classification) or mean (regression)
- Important hyperparameter: K
- Feature scaling affects performance (use StandardScaler)

---

## Next Steps

1. KNN Regression (average of neighbors)
2. Scaling features (Standardization/Normalization)
3. Compare KNN with other algorithms like Decision Trees

---

**End of Phase 4: K-Nearest Neighbors (KNN)**

