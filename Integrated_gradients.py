import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score
import sys

# %% HYPERPARAMETERS 

TRAIN = False
EPOCHS = 1000
BATCH_SIZE = 32
np.random.seed(0)

# L2_REG = 0
# ACTIV = 'tanh'
# LR = 1e-2
# DROP_RATE = 0
# UNITS = [128, 16, 128, 128, 64]

L2_REG = 1e-3
ACTIV = 'elu'
LR = 5e-4
DROP_RATE = 0.5
UNITS = [16, 16, 64, 16, 16]

# L2_REG = 1e-3
# ACTIV = 'relu'
# LR = 1e-2
# DROP_RATE = 0.0
# UNITS = [16, 16, 16, 16, 128]

# %% DATA PREPARATION 

# Load data
faults = pd.read_csv('Failure_Senja grid 12.csv')
feature_names = faults.columns[1:-1]
faults_clean = faults.dropna(axis='rows', how='any')
X = faults_clean.values[:,1:-1]
y = faults_clean.values[:,-1].astype(np.int)
X = np.concatenate((X, np.arange(X.shape[0])[...,None]), axis=-1) # keep track of the indices

# Subsample the non-fault class
sampling_rate=60
neg_idx = np.where(y==0)[0]
delete_idx = np.setdiff1d(neg_idx, neg_idx[::sampling_rate])
y = np.delete(y, delete_idx)
X = np.delete(X, delete_idx, axis=0)

# Shuffle and split the dataset
train_features, val_features, train_labels, val_labels = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=0)
train_idx, val_idx = train_features[:,-1].astype(np.int), val_features[:,-1].astype(np.int)
train_features, val_features = train_features[:,:-1], val_features[:,:-1]

# Normalize the features
scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features).astype(np.float32)

# Oversample the minority class in the training set
pos_features = train_features[train_labels==1]
neg_features = train_features[train_labels==0]
pos_labels = train_labels[train_labels==1]
neg_labels = train_labels[train_labels==0]

# Balance the dataset manually by choosing the right number of random indices from the positive examples
ids = np.arange(len(pos_features))
choices = np.random.choice(ids, len(neg_features))
res_pos_features = pos_features[choices] 
res_pos_labels = pos_labels[choices] 

# Concatenate the resampled positive samples with the negative samples
resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

# Shuffle the data
order = np.arange(len(resampled_labels))
np.random.shuffle(order)
resampled_features = resampled_features[order].astype(np.float32)
resampled_labels = resampled_labels[order]
print("resampled_features shape:", resampled_features.shape)

# Compute class proportions and weights in the validation set
neg, pos = np.bincount(val_labels)
val_labels_weights = np.copy(val_labels)
val_labels_weights[val_labels_weights==0] = pos
val_labels_weights[val_labels_weights==1] = neg

# # Add a fake class to use as baseline
# fake_size = int(np.median(np.bincount(y)))
# y_fake = np.ones(fake_size)*y.max()+1
# y = np.concatenate((y, y_fake))
# X_fake = np.zeros((fake_size, X.shape[-1]))
# X = np.concatenate((X, X_fake), axis=0)

# Number of classes
n_classes = len(np.unique(resampled_labels))

# %% Model definition and training
if TRAIN==True:

    METRICS = [
          # keras.metrics.FalsePositives(name='fp'),
          # keras.metrics.FalseNegatives(name='fn'), 
          # keras.metrics.AUC(name='prc', curve='PR'), 
          keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
    ]
    
    
    def make_model(n_classes, metrics=METRICS):
      model = keras.Sequential()
      for units in UNITS:
          
          model.add(keras.layers.Dense(units, activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2_REG)))
          model.add(keras.layers.BatchNormalization())
          model.add(keras.layers.Activation(ACTIV))
          model.add(keras.layers.Dropout(DROP_RATE))
          
      model.add(keras.layers.Dense(n_classes, activation=None))
    
      model.compile(
          optimizer=keras.optimizers.Adam(lr=LR),
           loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=metrics
          )
    
      return model
    
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            patience=30,
            mode='min',
            restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=0.5, 
            patience=10)
        ]
    
    
    model = make_model(n_classes)

    resampled_history = model.fit(
        resampled_features,
        resampled_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,    
        callbacks=callbacks,
        validation_data=(val_features, val_labels),
        )

    # Save the entire model as a SavedModel.
    model.save('saved_model/mlp_classifier')

else:
    model = tf.keras.models.load_model('saved_model/mlp_classifier_40')


# %% Prediction results of the model
y_probs = tf.nn.softmax(model(val_features), axis=-1).numpy()
y_pred = y_probs.argmax(axis=-1)
cm = confusion_matrix(val_labels, y_pred > 0.5)    
f1 = f1_score(val_labels, y_pred, sample_weight=val_labels_weights)

print("Results:\n    True Negatives: {:d}\n    False Positives: {:d}\n    False Negatives: {:d}\n    True Positives: {:d}\n    F1 score: {:.3f}\n ".format(
    cm[0][0], cm[0][1], cm[1][0], cm[1][1], f1 ))


FN_idx_orig = []
TP_idx_orig = []
FN_idx = []
TP_idx = []
FN_probs = []
TP_probs = []
FN_dates = []
TP_dates = []
for i, (yt, yprd, yprb, vid) in enumerate(zip(val_labels, y_pred, y_probs, val_idx)):
    if yt==1 and yprd==0:
        FN_idx_orig.append(vid)
        FN_idx.append(i)
        FN_probs.append(yprb.max())
        FN_dates.append(faults_clean["Date"].iloc[vid])
    elif yt==1 and yprd==1:
        TP_idx_orig.append(vid)
        TP_idx.append(i)
        TP_probs.append(yprb.max())
        TP_dates.append(faults_clean["Date"].iloc[vid])
        
FN_df = pd.DataFrame({"FN_idx": FN_idx, "FN_probs": FN_probs, "FN_idx_orig": FN_idx_orig, "Date": FN_dates})
TP_df = pd.DataFrame({"TP_idx": TP_idx, "TP_probs": TP_probs, "TP_idx_orig": TP_idx_orig, "Date": TP_dates})
print(FN_df)
print("")
print(TP_df)


# %% Select the sample to test with IGs
sample_idx = 52
sample_X = val_features[sample_idx]
sample_y = val_labels[sample_idx]

# %% Baseline selection
# The baseline should be an uninformative input, which gives equal probability to each class. 
# Models with structured data that typically involve a mix of continuous 
# numeric features will typically use the observed median value as a baseline 
# because 0 is an informative value for these features.

# Mean baseline
baseline_m = tf.constant(np.mean(resampled_features, axis=0))
logits_m = model(tf.expand_dims(baseline_m, axis=0))

# Zero baseline
baseline_z = tf.zeros(shape=train_features[0].shape)
logits_z = model(tf.expand_dims(baseline_z, axis=0))

# Random baseline
baseline_r = tf.random.uniform(shape=train_features[0].shape)
logits_r = model(tf.expand_dims(baseline_r, axis=0))

# fig = plt.figure(figsize=(10, 5))
# plt.subplot(2, 3, 1)
# plt.bar(np.arange(logits_m.shape[-1]), logits_m[0,:], color='orange')
# plt.gca().set_title('Mean baseline')
# plt.gca().set_xlabel('Logits')
# plt.gca().set_xticks([0,1])
# plt.subplot(2, 3, 4)
# plt.bar(np.arange(logits_m.shape[-1]), tf.nn.softmax(logits_m, axis=-1)[0,:], color='orange')
# plt.gca().set_ylim([0,1])
# plt.gca().set_xlabel('Class prob.')
# plt.gca().set_xticks([0,1])
# plt.subplot(2, 3, 2)
# plt.bar(np.arange(logits_z.shape[-1]), logits_z[0,:], color='orange')
# plt.gca().set_title('Zero baseline')
# plt.gca().set_xlabel('Logits')
# plt.gca().set_xticks([0,1])
# plt.subplot(2, 3, 5)
# plt.bar(np.arange(logits_z.shape[-1]), tf.nn.softmax(logits_z, axis=-1)[0,:], color='orange')
# plt.gca().set_ylim([0,1])
# plt.gca().set_xlabel('Class prob.')
# plt.gca().set_xticks([0,1])
# plt.subplot(2, 3, 3)
# plt.bar(np.arange(logits_r.shape[-1]), logits_r[0,:], color='orange')
# plt.gca().set_title('Random baseline')
# plt.gca().set_xlabel('Logits')
# plt.gca().set_xticks([0,1])
# plt.subplot(2, 3, 6)
# plt.bar(np.arange(logits_r.shape[-1]), tf.nn.softmax(logits_r, axis=-1)[0,:], color='orange')
# plt.gca().set_ylim([0,1])
# plt.gca().set_xlabel('Class prob.')
# plt.gca().set_xticks([0,1])
# plt.tight_layout()
# plt.show()

fig = plt.figure(figsize=(7, 3))
plt.subplot(1, 3, 1)
plt.bar(np.arange(logits_z.shape[-1]), tf.nn.softmax(logits_z, axis=-1)[0,:], color='orange')
plt.gca().set_title('Zero baseline')
plt.gca().set_ylim([0,1])
plt.gca().set_xlabel('Class prob.')
plt.gca().set_xticks([0,1])
plt.subplot(1, 3, 2)
plt.bar(np.arange(logits_m.shape[-1]), tf.nn.softmax(logits_m, axis=-1)[0,:], color='orange')
plt.gca().set_title('Mean baseline')
plt.gca().set_ylim([0,1])
plt.gca().set_xlabel('Class prob.')
plt.gca().set_xticks([0,1])
plt.subplot(1, 3, 3)
plt.gca().set_title('Random baseline')
plt.bar(np.arange(logits_r.shape[-1]), tf.nn.softmax(logits_r, axis=-1)[0,:], color='orange')
plt.gca().set_ylim([0,1])
plt.gca().set_xlabel('Class prob.')
plt.gca().set_xticks([0,1])
plt.tight_layout()
plt.savefig("baselines.pdf")
plt.show()


# Select the baseline to use
baseline = baseline_m


# Plot the baseline
fig = plt.figure(figsize=(3.5, 4))
ax = plt.gca()
df = pd.DataFrame({"feat names": feature_names, "values": baseline})
df.plot.bar(x="feat names", y="values", ax=ax, legend=False, color='k')
ax.set_xlabel("")
plt.title("Baseline")
plt.tight_layout()
plt.savefig("baseline.pdf")
plt.show()

# %% Check that the selected baseline works

# Generate m_steps intervals for integral_approximation() below.
m_steps=50
alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1) 

def interpolate_data(baseline,
                     data_sample,
                     alphas):
    alphas_x = alphas[:, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(data_sample, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    return images

interpolated_data = interpolate_data(
    baseline=baseline,
    data_sample=sample_X,
    alphas=alphas)

pred = model(interpolated_data)
pred_proba = tf.nn.softmax(pred, axis=-1)

# Plot how the prediction change as we move from the baseline to the actual features
fig = plt.figure(figsize=(15, 7))
i = 0
for alpha, sample, proba in zip(alphas[0::10], interpolated_data[0::10], pred_proba[0::10]):
    i += 1
    ax1 = plt.subplot(2, len(alphas[0::10]), i)
    plt.title(f'alpha: {alpha:.1f}')
    df1 = pd.DataFrame({"feat names": feature_names, "values": sample})
    df1.plot.bar(x="feat names", y="values", ax=ax1, legend=False)
    ax1.set_xlabel("")
    ax2 = plt.subplot(2, len(alphas[0::10]), i+len(alphas[0::10]))
    df2 = pd.DataFrame({"classes": ["non-fault", "fault"], "values": proba})
    df2.plot.bar(x="classes", y="values", ax=ax2, legend=False, color='orange')
    ax2.set_xlabel("")
    ax2.set_ylim([0,1])
plt.tight_layout()
plt.savefig("interp.pdf")
plt.show()


# The gradient tells us which input feats have the steepest local slope between our 
# output model's predicted class probabilities with respect to the original input feats.
def compute_gradients(samples, target_class_idx):
  with tf.GradientTape() as tape:
    tape.watch(samples)
    logits = model(samples)
    probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
  return tape.gradient(probs, samples)

# These gradients measure the change in our model's predictions 
# for each small step in the feature space starting from the baseline
path_gradients = compute_gradients(
    samples=interpolated_data,
    target_class_idx=sample_y)


plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(alphas, pred_proba[:, sample_y])
ax1.set_title('Target class predicted probability over alpha')
ax1.set_ylabel('model p(target class)')
ax1.set_xlabel('alpha')
ax1.set_ylim([0, 1])
ax2 = plt.subplot(1, 2, 2)
# Average across interpolation steps
average_grads = tf.reduce_mean(path_gradients, axis=-1)
# Normalize gradients to 0 to 1 scale. E.g. (x - min(x))/(max(x)-min(x))
average_grads_norm = (average_grads-tf.math.reduce_min(average_grads))/(tf.math.reduce_max(average_grads)-tf.reduce_min(average_grads))
ax2.plot(alphas, average_grads_norm)
ax2.set_title('Average feature gradients (normalized) over alpha')
ax2.set_ylabel('Average feature gradients')
ax2.set_xlabel('alpha')
ax2.set_ylim([0, 1]);
plt.tight_layout()
plt.savefig("gradients.pdf")
plt.show()


# %% Compute the IGs

def integral_approximation(gradients):
    # riemann_trapezoidal intergal approximation
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

ig = integral_approximation(
    gradients=path_gradients)

@tf.function # compiles the function into a high performance callable TensorFlow graph
def integrated_gradients(baseline,
                          sample,
                          target_class_idx,
                          m_steps=50,
                          batch_size=16):
    # 1. Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

    # Initialize TensorArray outside loop to collect gradients. This data structure
    # is similar to a Python list but more performant and supports backpropogation.
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        # 2. Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = interpolate_data(baseline=baseline,
                                                        data_sample=sample,
                                                        alphas=alpha_batch)

        # 3. Compute gradients between model outputs and interpolated inputs.
        gradient_batch = compute_gradients(samples=interpolated_path_input_batch,
                                        target_class_idx=target_class_idx)

        # Write batch indices and gradients to extend TensorArray.
        # Writing batch indices with scatter() allows for uneven batch sizes. 
        # Note: this operation is similar to a Python list extend().
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    
    
    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input.
    integrated_gradients = (sample - baseline) * avg_gradients

    return integrated_gradients


ig_attributions = integrated_gradients(baseline=baseline,
                                        sample=sample_X,
                                        target_class_idx=sample_y,
                                        m_steps=100)

# Negative values correspond to parts of the sample that if moved closer to the baseline value 
# would cause the prediction score to decrease. 
# In contrast postivie values correspond to parts of the image that if they were moved away from the baseline value, 
# the prediction score would increase.
fig = plt.figure(figsize=(7, 4))
ax1 = plt.subplot(1, 2, 1)
df1 = pd.DataFrame({"feat names": feature_names, "values": sample_X})
df1.plot.bar(x="feat names", y="values", ax=ax1, legend=False)
ax1.set_xlabel("")
ax1.set_title("Sample {} - {:s}".format(sample_idx, TP_df[TP_df["TP_idx"]==sample_idx]["Date"].values[0]))
ax2 = plt.subplot(1, 2, 2)
df2 = pd.DataFrame({"feat names": feature_names, "values": ig_attributions})
df2['positive'] = df2['values'] > 0
df2.plot.bar(x="feat names", y="values", ax=ax2, legend=False, color=df2.positive.map({True: 'g', False: 'r'}))
ax2.set_xlabel("")
ax2.set_title("Integrated gradients")
plt.tight_layout()
plt.savefig("IG_example.pdf")
plt.show()


def convergence_check(model, attributions, baseline, input, target_class_idx):
    """
    Args:
      model(keras.Model): A trained model to generate predictions and inspect.
      baseline(Tensor): A 3D image tensor with the shape 
        (image_height, image_width, 3) with the same shape as the input tensor.
      input(Tensor): A 3D image tensor with the shape 
        (image_height, image_width, 3).
      target_class_idx(Tensor): An integer that corresponds to the correct 
        ImageNet class index in the model's output predictions tensor. Default 
          value is 50 steps.   
    Returns:
      (none): Prints scores and convergence delta to sys.stdout.
    """
    # Your model's prediction on the baseline tensor. Ideally, the baseline score
    # should be close to zero.
    baseline_prediction = model(tf.expand_dims(baseline, 0))
    baseline_score = tf.nn.softmax(tf.squeeze(baseline_prediction))[target_class_idx]
    # Your model's prediction and score on the input tensor.
    input_prediction = model(tf.expand_dims(input, 0))
    input_score = tf.nn.softmax(tf.squeeze(input_prediction))[target_class_idx]
    # Sum of your IG prediction attributions.
    ig_score = tf.math.reduce_sum(attributions)
    delta = ig_score - (input_score - baseline_score)
    try:
        # Test your IG score is <= 5% of the input minus baseline score.
        tf.debugging.assert_near(ig_score, (input_score - baseline_score), rtol=0.05)
        tf.print('Approximation accuracy within 5%.', output_stream=sys.stdout)
    except tf.errors.InvalidArgumentError:
        tf.print('Increase or decrease m_steps to increase approximation accuracy.', output_stream=sys.stdout)
    
    tf.print('Baseline score: {:.3f}'.format(baseline_score))
    tf.print('Input score: {:.3f}'.format(input_score))
    tf.print('IG score: {:.3f}'.format(ig_score))     
    tf.print('Convergence delta: {:.3f}'.format(delta))
  
convergence_check(model=model,
                  attributions=ig_attributions, 
                  baseline=baseline, 
                  input=sample_X, 
                  target_class_idx=sample_y)

