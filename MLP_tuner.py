import tensorflow as tf
from tensorflow import keras
from kerastuner.tuners import RandomSearch, BayesianOptimization, Hyperband
import kerastuner
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

np.random.seed(0)

############################# DATA PREPARATION ###############################

# Load data
faults = pd.read_csv('Failure_Senja grid 12.csv')
faults_clean = faults.dropna(axis='rows', how='any')
X = faults_clean.values[:,1:-1].astype(np.float32)
y = faults_clean.values[:,-1].astype(np.int)
# feature_names = faults_clean.columns[1:-1]

# Subsample the non-fault class
sampling_rate=60
neg_idx = np.where(y==0)[0]
delete_idx = np.setdiff1d(neg_idx, neg_idx[::sampling_rate])
y = np.delete(y, delete_idx)
X = np.delete(X, delete_idx, axis=0)

# Examine the class imbalanceness
neg, pos = np.bincount(y)
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))

# Compute the class weights (scaling by total/2 helps keep the loss to a similar magnitude)
weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0
class_weight = {0: weight_for_0, 1: weight_for_1}
print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

# Split and shuffle the dataset
train_features, val_features, train_labels, val_labels = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True)

# Normalize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)


# Oversample the minority class in the training set
pos_features = train_features[train_labels==1]
neg_features = train_features[train_labels==0]
pos_labels = train_labels[train_labels==1]
neg_labels = train_labels[train_labels==0]

# Balance the dataset manually by choosing the right number of random indices from the positive examples
ids = np.arange(len(pos_features))
choices = np.random.choice(ids, len(neg_features))
res_pos_features = pos_features[choices] # Resampled training features
res_pos_labels = pos_labels[choices] # Resampled training labels

# Concatenate the resampled positive samples with the negative samples
resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

# Shuffle the data
order = np.arange(len(resampled_labels))
np.random.shuffle(order)
resampled_features = resampled_features[order]
resampled_labels = resampled_labels[order]

print("resampled_features shape:", resampled_features.shape)

# # One-hot encoding
# resampled_labels = OneHotEncoder(sparse=False).fit_transform(resampled_labels.reshape(-1,1)).astype(np.int)
# val_labels = OneHotEncoder(sparse=False).fit_transform(val_labels.reshape(-1,1)).astype(np.int)
# test_labels = OneHotEncoder(sparse=False).fit_transform(test_labels.reshape(-1,1)).astype(np.int)



############################# MODEL DEFINITION ###############################

# Metrics that are monitored during training
METRICS = [
      # keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      # keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      # keras.metrics.Precision(name='precision'),
      # keras.metrics.Recall(name='recall'),
      # keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_prc',
        verbose=1,
        patience=15,
        mode='max',
        restore_best_weights=True
        ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_prc',
        mode='max',
        factor=0.5, 
        patience=5)
    ]


def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', min_value=2, max_value=5)):
        model.add(keras.layers.Dense(units= hp.Choice('units_' + str(i), [16, 32, 64, 128]), #hp.Int('units_' + str(i), min_value=16, max_value=128, step=32),
                                     activation=None,
                                     kernel_regularizer=tf.keras.regularizers.l2(hp.Choice('l2_reg', [0.0, 1e-5, 1e-4, 1e-3]))
                                     )
                  )
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(hp.Choice('activation', ['relu', 'elu', 'tanh'])))
        model.add(keras.layers.Dropout(hp.Choice('dropout', [0.0, 0.1, 0.3, 0.5])))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=METRICS)
    
    return model


tuner = BayesianOptimization(
    build_model,
    objective=kerastuner.Objective("val_prc", direction="max"),
    max_trials=500,
    executions_per_trial=1,
    directory='keras_tuner',
    project_name='basic_MLP')

print(tuner.search_space_summary())

tuner.search(resampled_features, 
              resampled_labels,
              epochs=1000,
              batch_size=32,
              validation_data=(val_features, val_labels),
              callbacks=callbacks,
              verbose=2
              )

print(tuner.results_summary())
