import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

# Split dataset into training and testing sets
train, test = train_test_split(dataset, test_size=0.2, shuffle=False)
train_labels, test_labels = train_test_split(labels, test_size=0.2, shuffle=False)
print(len(train), 'train examples')
print(len(test), 'test examples')


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


# Final model parameters
if mode == 1:
    model = create_model(0.00001, 32, 2, "relu", "Adam", "glorot_uniform", 0.0, 0.0, 0.0)
    history = model.fit(train, train_labels, epochs=100, batch_size=8, shuffle=True, validation_split=0.2, verbose=1)

if mode == 2:
    model = create_model(0.001, 80, 3, "relu", "Adam", "glorot_uniform", 0.0, 0, 0)
    history = model.fit(train, train_labels, epochs=200, batch_size=128, shuffle=True, validation_split=0.2, verbose=2)

if mode == 3:
    model = create_model(0.00001, 64, 3, "elu", "Adam", "he_normal", 0.2, 0, 0)
    history = model.fit(train, train_labels, epochs=100, batch_size=64, shuffle=True, validation_split=0.2, verbose=2)


# Clamp values and set to int
if mode == 2:
    ynew = model.predict(test)
    ynew = np.rint(ynew)
    ynew[ynew < 0] = -1
    ynew[ynew > 18] = 18
    ynew = ynew.astype(int)

else:
    ynew = model.predict_classes(test)
    print(accuracy_score(test_labels, ynew))

print(model.summary())

# Save model
model.save("model" + str(mode) + ".h5")
print("Saved model to disk")
print()
print()

# Plot accuracy
if mode == 1:
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('Sparse Categorical Accuracy')
    plt.ylabel('Accuracy')

if mode == 2:
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('Mean Squared Error')
    plt.ylabel('Mean Squared Error')

if mode == 3:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')

# Plot loss
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plot confusion matrix
test_labels = test_labels.flatten()
ynew = ynew.flatten()
cm = confusion_matrix(test_labels, ynew)
if mode == 2:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(-1, (max_nodes*2-1)))
else:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
