import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

# Ensure reproducible results
np.random.seed(1)
tf.random.set_seed(1)

# Count files
cwd = os.path.dirname(os.path.realpath(__file__))
folder = os.path.join(cwd, "graphs")
list = os.listdir(folder)
number_files = len(list)
data = []

# Select mode
mode = input("Enter 1 to train for Connectivity, 2 to train for Degree Sequences, or 3 to train for Planarity. ")
while not (mode == "1" or mode == "2" or mode == "3"):
    mode = input("Enter 1 to train for Connectivity, 2 to train for Degree Sequences, or 3 to train for Planarity. ")

mode = int(mode)

# Loops for every file in the directory
# Takes the inputted JSON Graph and stores it in a dictionary
for i in range(number_files-1):
    f = open(os.path.join(folder, "graph"+str(i)+".json"))
    data.append(json.load(f))
    f.close()

f = open(os.path.join(folder, "config.json"))
max_nodes = json.load(f)['max_nodes']
f.close()

# Create placeholder array
# Max number of edges with the max number of nodes
# Pad array with -1
max_edges = max_nodes * (max_nodes - 1)
dataset = np.full((number_files-1, max_edges+max_nodes, 2), -1)
if mode == 2:
    labels = np.full((number_files-1, max_nodes), -1)
else:
    labels = np.full(number_files-1, -1)

# Transfer dictionaries to the numpy array
for i in range(number_files-1):
    if mode == 1:
        labels[i] = data[i]['type']
    elif mode == 2:
        temp = data[i]['degree']
        for l in range(len(data[i]['nodes'])):
            labels[i][l] = temp[l][1]
    elif mode == 3:
        labels[i] = data[i]['planarity']
    for j in range(len(data[i]['nodes'])):
        dataset[i][j][0] = data[i]['nodes'][j]
    for k in range(len(data[i]['edges'])):
        dataset[i][k+max_nodes] = data[i]['edges'][k]

# Undersample categorical data
if mode == 1 or mode == 3:
    undersample = RandomUnderSampler()
    dataset, labels = undersample.fit_resample(dataset.reshape(-1, (max_edges + max_nodes) * 2), labels)

# Reshape data after undersample
dataset = dataset.reshape(-1, max_edges+max_nodes, 2)
if mode == 2:
    labels = labels.reshape(-1, max_nodes)

# Shuffle array with saved state to ensure samples match labels
rng_state = np.random.get_state()
np.random.shuffle(dataset)
np.random.set_state(rng_state)
np.random.shuffle(labels)


# Create neural network model
def create_model(lr, neurons, layers, activation, optimizer, initializer, dropout, l1, l2):
    model = tf.keras.Sequential()
    # Ensure data structure
    model.add(tf.keras.layers.Reshape((max_edges+max_nodes, 2)))
    # Iterate for number of layers
    for i in range(layers):
        # Create dense layer with specified parameters
        x = tf.keras.layers.Dense(neurons, activation=activation, kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2))
        model.add(x)
        # Create dropout layer with specified dropout rate
        y = tf.keras.layers.Dropout(dropout)
        model.add(y)
    model.add(tf.keras.layers.Flatten()),
    # Set output layer, loss function, and metric
    if mode == 1:
        model.add(tf.keras.layers.Dense(4))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['sparse_categorical_accuracy']
    elif mode == 2:
        model.add(tf.keras.layers.Dense(max_nodes))
        loss = tf.keras.losses.MeanSquaredError(reduction="auto")
        metrics = ['mean_squared_error']
    elif mode == 3:
        model.add(tf.keras.layers.Dense(1))
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = ['accuracy']
    # Set optimization algorithm
    if optimizer == "Adam":
        alg = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == "SGD":
        alg = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == "RMSProp":
        alg = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif optimizer == "Adadelta":
        alg = tf.keras.optimizers.Adadelta(learning_rate=lr)
    elif optimizer == "Adamax":
        alg = tf.keras.optimizers.Adamax(learning_rate=lr)
    elif optimizer == "Adagrad":
        alg = tf.keras.optimizers.Adagrad(learning_rate=lr)
    elif optimizer == "Nadam":
        alg = tf.keras.optimizers.Nadam(learning_rate=lr)
    elif optimizer == "Ftrl":
        alg = tf.keras.optimizers.Ftrl(learning_rate=lr)
    # Compile model
    model.compile(optimizer=alg,
                  loss=loss,
                  metrics=metrics)
    return model


# Wrap model for sklearn
if mode == 2:
    model = KerasRegressor(build_fn=create_model, verbose=2, shuffle=True)
else:
    model = KerasClassifier(build_fn=create_model, verbose=2, shuffle=True)

# Search parameters for each model
if mode == 1:
    batch_size = [4096]
    epochs = [100]
    lr = [0.01, 0.001, 0.0001]
    neurons = [8, 16, 32]
    layers = [1, 2, 3]
    activation = ["relu"]
    optimizer = ["Adam"]
    initializer = ["glorot_uniform"]
    dropout = [0.2]
    l1 = [0]
    l2 = [0]

elif mode == 2:
    batch_size = [4096]
    epochs = [100]
    lr = [0.01, 0.001, 0.0001]
    neurons = [8, 16, 32]
    layers = [1, 2, 3]
    activation = ["relu"]
    optimizer = ["Adam"]
    initializer = ["glorot_uniform"]
    dropout = [0.2]
    l1 = [0]
    l2 = [0]

elif mode == 3:
    batch_size = [4096]
    epochs = [100]
    lr = [0.01, 0.001, 0.0001]
    neurons = [8, 16, 32]
    layers = [1, 2, 3]
    activation = ["relu"]
    optimizer = ["Adam"]
    initializer = ["glorot_uniform"]
    dropout = [0.2]
    l1 = [0]
    l2 = [0]

# Create parameter grid
param_grid = dict(batch_size=batch_size, epochs=epochs, lr=lr, neurons=neurons, layers=layers, activation=activation, optimizer=optimizer, initializer=initializer, dropout=dropout, l1=l1, l2=l2)
# Set scoring metric
if mode == 2:
    scoring = "neg_mean_squared_error"
else:
    scoring = "accuracy"
# Begin grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=1, cv=2, refit=False)
grid_result = grid.fit(dataset, labels)

# Print grid search results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Save grid search results to file
print(grid.cv_results_)
results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
out = results.to_json(orient='records', indent=4)

with open('results'+str(mode)+'.txt', 'w') as f:
    f.write(out)

