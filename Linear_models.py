import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score


# Load data
faults = pd.read_csv('Failure_Senja grid.csv')
faults_clean = faults.dropna(axis='rows', how='any')
X = faults_clean.values[:,1:5].astype(np.float32)
y = faults_clean.values[:,5].astype(np.int)
feature_names = faults_clean.columns[1:5]

# Subsample the non-fault class
sampling_rate=6
neg_idx = np.where(y==0)[0]
delete_idx = np.setdiff1d(neg_idx, neg_idx[::sampling_rate])
X = np.delete(X, delete_idx, axis=0)
y = np.delete(y, delete_idx)

# Standardize feats
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean)/X_std

#Train/test split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y)

models = [RidgeClassifier(class_weight='balanced'), LogisticRegression(class_weight='balanced'), SVC(kernel='linear', class_weight='balanced')]
model_names = ["RidgeClass", "LogisticReg", "LinSVC"]

res = []
for model, name in zip(models, model_names):
    # Coefficients of a linear model
    if len(model.fit(X,y).coef_.shape) > 1:
        importance = np.abs(model.coef_[0])
    else:
        importance = np.abs(model.coef_)
    res.append(importance)
    
    # Compute metrics
    y_pred = model.fit(Xtr, ytr).predict(Xte)
    cm = confusion_matrix(yte, y_pred > 0.5)    
    f1 = f1_score(yte, y_pred)

    print("{}:\n    True Negatives: {:d}\n    False Positives: {:d}\n    False Negatives: {:d}\n    True Positives: {:d}\n    F1 score: {:.3f}\n ".format(
        name, cm[0][0], cm[0][1], cm[1][0], cm[1][1], f1  ))

weights = np.vstack(res)
weights = weights/weights.sum(axis=1)[:,None]

df = pd.DataFrame(weights, columns=feature_names, index=model_names)
df.T.plot(kind='bar',alpha=0.75, rot=30, figsize=(8,4), title="Normalized magnitude of the cefficients")
plt.tight_layout()
plt.savefig("coeff_score.pdf")
