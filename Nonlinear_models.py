import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM, SVC
from sklearn.neural_network import MLPClassifier
from time import time
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Specify model and if we want to do feat selection
model_name = "OCSVM"
cross_validation = False

# Load data
faults = pd.read_csv('Failure_Senja grid 12.csv')
faults_clean = faults.dropna(axis='rows', how='any')
X = faults_clean.values[:,1:-1].astype(np.float32)
y = faults_clean.values[:,-1].astype(np.int)
feature_names = faults_clean.columns[1:-1]

# Subsample the non-fault class
sampling_rate=60
neg_idx = np.where(y==0)[0]
delete_idx = np.setdiff1d(neg_idx, neg_idx[::sampling_rate])
X = np.delete(X, delete_idx, axis=0)
y = np.delete(y, delete_idx)

# Train/test split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y)

# Normalize the features
scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)

# Compute class proportions and weights
neg, pos = np.bincount(ytr)
yte_weights = np.copy(yte)
yte_weights[yte_weights==0] = pos
yte_weights[yte_weights==1] = neg


if model_name == "MLP":
    model = MLPClassifier(hidden_layer_sizes=(32,32), max_iter=500, activation='relu', batch_size=200, solver='adam', alpha=1e-2)
elif model_name == "SVC":
    model = SVC(kernel='rbf', class_weight='balanced')
elif model_name == "OCSVM":
    model = OneClassSVM(kernel='rbf', nu=0.1, gamma='auto')
    # y = y*2-1

# Compute metrics
y_pred = model.fit(Xtr, ytr).predict(Xte)
if model_name == "OCSVM":
    y_pred = (y_pred+1)/2
cm = confusion_matrix(yte, y_pred)    
f1 = f1_score(yte, y_pred, sample_weight=yte_weights)
print("{}:\n    True Negatives: {:d}\n    False Positives: {:d}\n    False Negatives: {:d}\n    True Positives: {:d}\n    F1 score: {:.3f}\n ".format(
        model_name, cm[0][0], cm[0][1], cm[1][0], cm[1][1], f1  ))


################## Greedy selection based on cross-validation
if cross_validation:
    tic_fwd = time()
    sfs_forward = SequentialFeatureSelector(model, scoring='f1', 
                                            n_features_to_select=3,
                                            direction='forward',
                                            cv=5,
                                            n_jobs=-1
                                            ).fit(X, y)
    toc_fwd = time()
    print("Features selected by forward sequential selection: "
          f"{feature_names[sfs_forward.get_support()]}")
    print(f"Done in {toc_fwd - tic_fwd:.3f}s")
    
    tic_bwd = time()
    sfs_backward = SequentialFeatureSelector(model, scoring='f1', 
                                              n_features_to_select=3,
                                              direction='backward', 
                                              cv=5,
                                              n_jobs=-1
                                              ).fit(X, y)
    toc_bwd = time()
    print("Features selected by backward sequential selection: "
          f"{feature_names[sfs_backward.get_support()]}")
    print(f"Done in {toc_bwd - tic_bwd:.3f}s")
