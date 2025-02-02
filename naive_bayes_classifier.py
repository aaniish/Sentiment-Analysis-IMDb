"""
hw05_part1.py (15%)
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
import joblib
import zipfile

# Load training data and unzip
with zipfile.ZipFile("train.data.zip", "r") as zip_ref:
    with zip_ref.open("train.data") as f:
        data = np.loadtxt(f, dtype=float)
X_train = data[:, :-1]
y_train = data[:, -1]

# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(clf, "model1.obj")
