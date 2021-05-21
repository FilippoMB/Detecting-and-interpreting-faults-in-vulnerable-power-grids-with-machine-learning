import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# Load data
faults = pd.read_csv('Failure_Senja grid.csv')
faults_clean = faults.dropna(axis='rows', how='any')
X = faults_clean.values[:,1:5].astype(np.float32)
y = faults_clean.values[:,5].astype(np.int)
feature_names = faults_clean.columns[1:5]

# Subsample the non-fault class
sampling_rate=27
neg_idx = np.where(y==0)[0]
delete_idx = np.setdiff1d(neg_idx, neg_idx[::sampling_rate])
X = np.delete(X, delete_idx, axis=0)
y = np.delete(y, delete_idx)

f1_list = []
tn_list = []
fp_list = []
fn_list = []
tp_list = []
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    
    # Train/test split
    Xtr = X[train_index]
    ytr = y[train_index]
    Xte = X[test_index]
    yte = y[test_index]
    
    # Train/validation split (the validation can be used to set the hyperparameters and will be used in the MLP to perform the early stop)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr)

    # Compute class proportions and weights
    neg, pos = np.bincount(ytr)
    yte_weights = np.copy(yte)
    yte_weights[yte_weights==0] = pos
    yte_weights[yte_weights==1] = neg
    
    # Normalize the features
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xval = scaler.transform(Xval)
    Xte = scaler.transform(Xte)

    # Specify/initialize the model
    model = LogisticRegression(class_weight='balanced') # RidgeClassifier(class_weight='balanced') SVC(kernel='linear', class_weight='balanced')

    # Fit/train the model
    model.fit(Xtr, ytr)
    
    # Compute predictions
    y_pred = model.predict(Xte)

    # Compute metrics
    cm = confusion_matrix(yte, y_pred > 0.5)    
    f1 = f1_score(yte, y_pred, sample_weight=yte_weights)

    print("Results fold {:d}:\n    True Negatives: {:d}\n    False Positives: {:d}\n    False Negatives: {:d}\n    True Positives: {:d}\n    F1 score: {:.3f}\n ".format(
        i, cm[0][0], cm[0][1], cm[1][0], cm[1][1], f1 ))

    f1_list.append(f1)
    tn_list.append(cm[0][0])
    fp_list.append(cm[0][1])
    fn_list.append(cm[1][0])
    tp_list.append(cm[1][1])


print("Final results (averaged):\n    True Negatives: {:d}\n    False Positives: {:d}\n    False Negatives: {:d}\n    True Positives: {:d}\n    F1 score: {:.3f}\n ".format(
        int(np.mean(tn_list)), int(np.mean(fp_list)), int(np.mean(fn_list)), int(np.mean(tp_list)), np.mean(f1_list) ))