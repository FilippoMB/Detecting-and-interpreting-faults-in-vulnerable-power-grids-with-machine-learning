import tensorflow as tf
from tensorflow import keras
from kerastuner.tuners import RandomSearch, BayesianOptimization, Hyperband
import kerastuner
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection

np.random.seed(0)

############################# DATA PREPARATION ###############################

# Load data
faults = pd.read_csv('Failure_Senja grid 12.csv')
faults_clean = faults.dropna(axis='rows', how='any')
X = faults_clean.values[:,1:-1].astype(np.float32)
Y = faults_clean.values[:,-1].astype(np.int)

# Subsample the non-fault class
sampling_rate=60
neg_idx = np.where(Y==0)[0]
delete_idx = np.setdiff1d(neg_idx, neg_idx[::sampling_rate])
Y = np.delete(Y, delete_idx)
X = np.delete(X, delete_idx, axis=0)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

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
        model.add(keras.layers.Dense(units= hp.Choice('units_' + str(i), [16, 32, 64, 128]), 
                                     activation=None,
                                     kernel_regularizer=tf.keras.regularizers.l2(hp.Choice('l2_reg', [0.0, 1e-5, 1e-4, 1e-3]))
                                     )
                  )
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(hp.Choice('activation', ['relu', 'elu', 'tanh'])))
        model.add(keras.layers.Dropout(hp.Float('dropout', min_value=0.0, max_value=0.9, step=0.1)))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=METRICS)
    
    return model


class CVTuner(kerastuner.engine.multi_execution_tuner.MultiExecutionTuner):
    
    def __init__(self, oracle, hypermodel, executions_per_trial, *args, **kwargs):
        super().__init__(oracle=oracle, 
                         hypermodel=hypermodel, 
                         executions_per_trial=executions_per_trial, 
                         *args, **kwargs)
    
    def run_trial(self, trial, x, y, n_folds=2, **kwargs):
        cv = model_selection.KFold(n_folds, random_state=0, shuffle=True)
        val_metrics = []
        for train_indices, val_indices in cv.split(x):
            x_train, x_val = x[train_indices], x[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            
            # Balance the dataset manually by oversampling the positive class
            pos_x = x_train[y_train==1]
            neg_x = x_train[y_train==0]
            pos_y = y_train[y_train==1]
            neg_y = y_train[y_train==0]
            ids = np.arange(len(pos_x))
            choices = np.random.choice(ids, len(neg_x))
            res_pos_x = pos_x[choices] # Resampled positive training features
            res_pos_y = pos_y[choices] # Resampled positive training labels
            
            # Concatenate the resampled positive samples with the negative samples
            resampled_x = np.concatenate([res_pos_x, neg_x], axis=0)
            resampled_y = np.concatenate([res_pos_y, neg_y], axis=0)
            
            # Shuffle and update the data
            order = np.arange(len(resampled_y))
            np.random.shuffle(order)
            x_train = resampled_x[order]
            y_train = resampled_y[order]
            
            # Build, train, and evaluate the model
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(x_train, y_train, validation_data=(x_val, y_val), **kwargs)
            val_metrics.append(model.evaluate(x_val, y_val))
        
        # Save average results across the different folds
        val_metrics = np.stack(val_metrics, axis=1).mean(axis=-1)
        val_metrics_names = ["val_"+n for n in model.metrics_names]
        self.oracle.update_trial(trial.trial_id, dict(zip(val_metrics_names, val_metrics)))
        self.save_model(trial.trial_id, model)


tuner = CVTuner(
    hypermodel=build_model,
    oracle=kerastuner.oracles.BayesianOptimization(
        objective=kerastuner.Objective("val_prc", direction="max"),
        max_trials=5000),
    executions_per_trial=1, # doesn't work to set it greater than 1, meaning that it does only 1 trial for each fold. this can be fixed by implementing the different runs in the "run_trial" function
    directory='keras_tuner',
    project_name='basic_MLP_cv')

print(tuner.search_space_summary())

tuner.search(X, Y,
             n_folds=5,
             epochs=1000,
             batch_size=32,
             callbacks=callbacks,
             verbose=2)

print(tuner.results_summary())
