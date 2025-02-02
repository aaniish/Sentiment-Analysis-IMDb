"""
hw05_part2.py (50%)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
import zipfile

# Load training data and unzip
with zipfile.ZipFile("train.data.zip", "r") as zip_ref:
    with zip_ref.open("train.data") as f:
        data = np.loadtxt(f, dtype=float)
X_train = data[:, :-1]
y_train = data[:, -1]

# Standardize the features
X_train = tf.keras.utils.normalize(X_train, axis=1)

# Design and fine-tune an ANN implemented using a Keras Sequential model on the training set
def build_model(hp):
    model = tf.keras.Sequential()

    # Input layer
    model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))

    # Tune the number of hidden layers and the number of nuerons of each hidden layer (units)
    for i in range(hp.Int("num_hidden_layers", 1, 5)):  
        model.add(layers.Dense(units=hp.Int("units_" + str(i), 16, 256, step=16), activation="relu"))  

    # Output layer
    model.add(layers.Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])

    return model


tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=15,  
    directory="tuner_results",
)

# Define an early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

tuner.search(X_train, y_train, epochs=10, validation_split=0.1, batch_size=32, callbacks=[early_stopping])

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Save the best model
best_model.save("model2.obj")
