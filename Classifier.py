import json
import keras.models
import numpy as np

# Select mode
mode = input("Enter 1 to predict Connectivity, 2 to predict Degree Sequences, or 3 to predict Planarity. ")
while not (mode == "1" or mode == "2" or mode == "3"):
    mode = input("Enter 1 to predict Connectivity, 2 to predict Degree Sequences, or 3 to predict Planarity. ")

mode = int(mode)

# Load model and file
model = keras.models.load_model("model" + str(mode) + ".h5")
f = open(r"graph.json")
sample = json.load(f)
f.close()

# Preprocess data
data = np.full((1, 100, 2), -1)
for i in range(len(sample["nodes"])):
    data[0][i][0] = sample["nodes"][i]
for j in range(len(sample["edges"])):
    data[0][j+10] = sample["edges"][j]

# Predict
if mode == 2:
    prediction = model.predict(data)
    prediction = np.rint(prediction)
    prediction[prediction < 0] = -1
    prediction[prediction > 18] = 18
    prediction = prediction.astype(int)
else:
    prediction = model.predict_classes(data)

print()
print()
# Print prediction
print("True value")
if mode == 1:
    print(sample["type"])
elif mode == 2:
    print(np.array(sample["degree"])[:, 1])
elif mode == 3:
    print(sample["planarity"])

print("Predicted value")
print(prediction)
