"""
hw05_part3.py (15%)
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import tensorflow.keras as keras
import joblib

# Load test data
data = np.loadtxt("test.data", dtype=float)
X_test = data[:, :-1]
y_test = data[:, -1]

# Standardize the features
X_test = keras.utils.normalize(X_test, axis=1)

# Load models
model1 = joblib.load("model1.obj")
model2 = keras.models.load_model("model2.obj")

# Make predictions
y_pred1 = model1.predict(X_test)
y_pred2 = (model2.predict(X_test) > 0.5).astype("int32")

# Calculate metrics
precision1 = precision_score(y_test, y_pred1)
recall1 = recall_score(y_test, y_pred1)
accuracy1 = accuracy_score(y_test, y_pred1)

precision2 = precision_score(y_test, y_pred2)
recall2 = recall_score(y_test, y_pred2)
accuracy2 = accuracy_score(y_test, y_pred2)

# Print metrics
print("Model 1 (Naive Bayes)")
print("Precision:", precision1)
print("Recall:", recall1)
print("Accuracy:", accuracy1)
print(f"Error rate: {(1 - accuracy1):.4f}")

print("\nModel 2 (ANN)")
print("Precision:", precision2)
print("Recall:", recall2)
print("Accuracy:", accuracy2)
print(f"Error rate: {(1 - accuracy2):.4f}")
