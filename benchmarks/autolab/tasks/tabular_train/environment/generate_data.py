"""
Generate a noisy tabular classification dataset and save train/test splits.

Design intent:
  - 200 features: 20 informative, 10 redundant, 170 pure noise
  - Features 100-199 are scaled up 20x so unnormalized models struggle with gradient flow
  - 5% label noise to make the problem realistically imperfect
  - 6000 train / 2000 test (stratified)
"""
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

os.makedirs("/data", exist_ok=True)
os.makedirs("/tests", exist_ok=True)

X, y = make_classification(
    n_samples=8000,
    n_features=200,
    n_informative=20,
    n_redundant=10,
    n_repeated=0,
    n_classes=2,
    weights=[0.58, 0.42],
    flip_y=0.05,
    class_sep=1.0,
    random_state=42,
)

# Scale the second half of features up 20x — penalises models that skip normalisation
X[:, 100:] *= 20.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y,
)

np.save("/data/X_train.npy", X_train.astype(np.float32))
np.save("/data/y_train.npy", y_train.astype(np.int64))
np.save("/tests/X_test.npy",  X_test.astype(np.float32))
np.save("/tests/y_test.npy",  y_test.astype(np.int64))

print(f"train={len(X_train)} test={len(X_test)} features={X.shape[1]} "
      f"pos_rate={y_train.mean():.3f}")
