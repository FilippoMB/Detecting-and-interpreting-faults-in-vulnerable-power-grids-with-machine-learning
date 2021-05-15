import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
# from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.regularizers import l2
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator
# from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector


faults = pd.read_csv('Failure_Senja grid.csv')
faults_drop = faults.dropna(axis='rows', how='any')

X = faults_drop.values[:,1:5].astype(np.float32)
y = faults_drop.values[:,5].astype(np.int)
feature_names = faults_drop.columns[1:5]

# Train/test split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y)

# Normalize the features
scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def make_model(metrics=METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(32, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                         input_shape=(Xtr.shape[-1],)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(32, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4),),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias),
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  return model


class custom_estimator(BaseEstimator):
    
    def __init__(self, model):
        
        self.model = model
        
        
    def fit(self, X, y, **kwargs):
        
        self.model.fit(X, y, **kwargs)
        
    
    

model = make_model()
est = custom_estimator(model)
# check_estimator(est)
# est.fit(X,y, epochs=5, batch_size=1024)

sfs_forward = SequentialFeatureSelector(est, scoring='f1', 
                                            n_features_to_select=3,
                                            direction='forward',
                                            cv=5,
                                            n_jobs=-1
                                            ).fit(X, y)