import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score

mpl.rcParams['figure.figsize'] = (10, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

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
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True)
train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.2, stratify=train_labels, shuffle=True)

# Normalize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)
print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)
print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

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

EPOCHS = 1000
BATCH_SIZE = 32
L2_REG = 0.0
ACTIV = 'tanh'
LR = 5e-3
DROP_RATE = 0.5
UNITS = [96, 16, 48]

# Metrics that are monitored during training
METRICS = [
      # keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      # keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      # keras.metrics.BinaryAccuracy(name='accuracy'),
      # keras.metrics.Precision(name='precision'),
      # keras.metrics.Recall(name='recall'),
      # keras.metrics.AUC(name='auc'),
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
      # loss=keras.losses.CategoricalCrossentropy(),
       metrics=metrics
      # metrics = keras.metrics.CategoricalAccuracy(name='accuracy'),
      )

  return model


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # monitor='val_loss',
        monitor='val_prc',
        verbose=1,
        patience=30,
        # mode='min',
        mode='max',
        restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(
        # monitor="val_loss", 
        monitor='val_prc',
        mode='max',
        factor=0.5, 
        patience=10)
    ]


model = make_model()
resampled_history = model.fit(
    resampled_features,
    resampled_labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,    
    callbacks=callbacks,
    validation_data=(val_features, val_labels),
    # class_weight=class_weight
    class_weight={0: 1, 1: 1}
    )


# Print the value of the metrics at the end of the training
results = model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, results):
  print(name, ': ', value)
print()

############################# PLOTS ###############################

# Function to plot the confusion matrix
def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])
    print('Total faults: ', np.sum(cm[1]))
    plt.show()
  
  
# Confusion matrix
test_predictions = model.predict(test_features, batch_size=BATCH_SIZE)
# plot_cm(test_labels.argmax(axis=-1), test_predictions.argmax(axis=-1), p=0.5)
plot_cm(test_labels, test_predictions, p=0.5)


# Function that plots how the metrics on training and validation set change during the training
def plot_metrics(history):
    metrics = ['loss', 'prc',] #'prc', 'precision', 'recall'
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(1,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.legend()
    plt.show()
    
# Make the plots
plot_metrics(resampled_history)
