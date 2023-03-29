import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras

################################ LOAD DATA ###################################
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

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

############################# MODEL DEFINITION ###############################

EPOCHS = 1000
BATCH_SIZE = 32

# L2_REG = 1e-3
# ACTIV = 'elu'
# LR = 5e-4
# DROP_RATE = 0.5
# UNITS = [16, 16, 64, 16, 16]

L2_REG = 1e-3
ACTIV = 'relu'
LR = 1e-2
DROP_RATE = 0.0
UNITS = [16, 16, 16, 16, 128]

# L2_REG = 1e-4
# ACTIV = 'relu'
# LR = 1e-2
# DROP_RATE = 0.0
# UNITS = [16, 32, 32, 32, 32]


# Metrics that are monitored during training
METRICS = [
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.FalseNegatives(name='fn'), 
    keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


def make_model(metrics=METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential()
  for units in UNITS:
      
      model.add(keras.layers.Dense(units, activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2_REG)))
      model.add(keras.layers.BatchNormalization())
      model.add(keras.layers.Activation(ACTIV))
      model.add(keras.layers.Dropout(DROP_RATE))
      
  model.add(keras.layers.Dense(1, activation='sigmoid'))

  model.compile(
      optimizer=keras.optimizers.Adam(lr=LR),
       loss=keras.losses.BinaryCrossentropy(),
       metrics=metrics
      )

  return model


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_prc',
        verbose=1,
        patience=30,
        mode='max',
        restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_prc',
        mode='max',
        factor=0.5, 
        patience=10)
    ]


################################## K-FOLD ####################################

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
    
    # Balance the dataset manually by oversampling the positive class
    pos_x = train_features[train_labels==1]
    neg_x = train_features[train_labels==0]
    pos_y = train_labels[train_labels==1]
    neg_y = train_labels[train_labels==0]
    ids = np.arange(len(pos_x))
    choices = np.random.choice(ids, len(neg_x))
    res_pos_x = pos_x[choices] # Resampled positive training features
    res_pos_y = pos_y[choices] # Resampled positive training labels
    
    # Concatenate the resampled positive samples with the negative samples
    resampled_features = np.concatenate([res_pos_x, neg_x], axis=0)
    resampled_labels = np.concatenate([res_pos_y, neg_y], axis=0)
    
    # Shuffle and update the data
    order = np.arange(len(resampled_labels))
    np.random.shuffle(order)
    train_features = resampled_features[order]
    train_labels = resampled_labels[order]
    

    # Specify/initialize the model
    model = make_model()

    # Fit/train the model
    resampled_history = model.fit(
        resampled_features,
        resampled_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,    
        callbacks=callbacks,
        validation_data=(val_features, val_labels),
    )
    
    # Compute predictions
    y_pred = model.predict(val_features)
    y_pred = np.round(y_pred)[:,0]

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