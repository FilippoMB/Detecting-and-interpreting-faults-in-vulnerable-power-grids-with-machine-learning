import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# Load data
faults = pd.read_csv('Failure_Senja grid 12.csv')
faults_clean = faults.dropna(axis='rows', how='any')
X = faults_clean.values[:,1:-1].astype(np.float32)
y = faults_clean.values[:,-1].astype(np.int)

# Subsample the non-fault class
sampling_rate=60
neg_idx = np.where(y==0)[0]
delete_idx = np.setdiff1d(neg_idx, neg_idx[::sampling_rate])
X = np.delete(X, delete_idx, axis=0)
y = np.delete(y, delete_idx)

f1_list = []
tn_list = []
fp_list = []
fn_list = []
tp_list = []
skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

for i, (train_index, val_index) in enumerate(skf.split(X, y)):
    
    # Train/test split
    train_features = X[train_index]
    train_labels = y[train_index]
    val_features = X[val_index]
    val_labels = y[val_index]
    
    # Compute class proportions and weights in the validation set
    neg, pos = np.bincount(val_labels)
    val_labels_weights = np.copy(val_labels)
    val_labels_weights[val_labels_weights==0] = pos
    val_labels_weights[val_labels_weights==1] = neg
    
    # Normalize the features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    # Specify/initialize the model
    model = model = SVC(kernel='linear', class_weight='balanced') #SVC(kernel='rbf', C=0.5, class_weight='balanced') # LogisticRegression(class_weight='balanced')  RidgeClassifier(class_weight='balanced')

    # Fit/train the model
    model.fit(train_features, train_labels)
    
    # Compute predictions
    y_pred = model.predict(val_features)

    # Compute metrics
    cm = confusion_matrix(val_labels, y_pred > 0.5)    
    f1 = f1_score(val_labels, y_pred, sample_weight=val_labels_weights)

    print("Results fold {:d}:\n    True Negatives: {:d}\n    False Positives: {:d}\n    False Negatives: {:d}\n    True Positives: {:d}\n    F1 score: {:.3f}\n ".format(
        i, cm[0][0], cm[0][1], cm[1][0], cm[1][1], f1 ))

    f1_list.append(f1)
    tn_list.append(cm[0][0])
    fp_list.append(cm[0][1])
    fn_list.append(cm[1][0])
    tp_list.append(cm[1][1])


print("Final results (averaged):\n    True Negatives: {:.1f}\n    False Positives: {:.1f}\n    False Negatives: {:.1f}\n    True Positives: {:.1f}\n    F1 score: {:.3f}\n ".format(
        np.mean(tn_list), np.mean(fp_list), np.mean(fn_list), np.mean(tp_list), np.mean(f1_list) ))