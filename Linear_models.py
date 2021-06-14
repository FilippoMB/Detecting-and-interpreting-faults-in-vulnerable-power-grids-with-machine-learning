import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import csv

np.random.seed(0)

# Load data
faults = pd.read_csv('Failure_Senja grid 12.csv')
faults_clean = faults.dropna(axis='rows', how='any')
X = faults_clean.values[:,1:-1].astype(np.float32)
y = faults_clean.values[:,-1].astype(np.int)
feature_names = faults_clean.columns[1:-1]
X = np.concatenate((X, np.arange(X.shape[0])[...,None]), axis=-1) # keep track of the indices

# Subsample the non-fault class
sampling_rate=60
neg_idx = np.where(y==0)[0]
delete_idx = np.setdiff1d(neg_idx, neg_idx[::sampling_rate])
X = np.delete(X, delete_idx, axis=0)
y = np.delete(y, delete_idx)


#Train/test split
train_features, val_features, train_labels, val_labels = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=0)
train_idx, val_idx = train_features[:,-1].astype(np.int), val_features[:,-1].astype(np.int)
train_features, val_features = train_features[:,:-1], val_features[:,:-1]

# Compute class proportions and weights
neg, pos = np.bincount(train_labels)
val_labels_weights = np.copy(val_labels)
val_labels_weights[val_labels_weights==0] = pos
val_labels_weights[val_labels_weights==1] = neg

# Normalize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)

models = [RidgeClassifier(class_weight='balanced'), LogisticRegression(class_weight='balanced'), SVC(kernel='linear', class_weight='balanced')]
model_names = ["RidgeClass", "LogisticReg", "LinSVC"]

res = []
for model, name in zip(models, model_names):
    # Coefficients of a linear model
    if len(model.fit(train_features, train_labels).coef_.shape) > 1:
        importance = np.abs(model.coef_[0])
    else:
        importance = np.abs(model.coef_)
    res.append(importance)
    
    # Compute metrics
    y_pred = model.predict(val_features)
    cm = confusion_matrix(val_labels, y_pred > 0.5)    
    f1 = f1_score(val_labels, y_pred, sample_weight=val_labels_weights)

    print("{}:\n    True Negatives: {:d}\n    False Positives: {:d}\n    False Negatives: {:d}\n    True Positives: {:d}\n    F1 score: {:.3f}\n ".format(
        name, cm[0][0], cm[0][1], cm[1][0], cm[1][1], f1 ))
    
    false_neg_idx = []
    true_pos_idx = []
    for yt, yp, vid in zip(val_labels, y_pred, val_idx):
        if yt==1 and yp==0:
            false_neg_idx.append(vid)
        elif yt==1 and yp==1:
            true_pos_idx.append(vid)
    print("false_neg_idx", false_neg_idx)
    print("true_pos_idx", true_pos_idx)
    # with open(r'false_neg.csv', 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(false_neg_idx)

weights = np.vstack(res)
weights = weights/weights.sum(axis=1)[:,None]

df = pd.DataFrame(weights, columns=feature_names, index=model_names)
df.T.plot(kind='bar',alpha=0.75, rot=30, figsize=(8,4), title="Normalized magnitude of the cefficients")
plt.tight_layout()
plt.savefig("coeff_score.pdf")
