# builds, trains, and saves a Keras neural network classifier on Fashion-MNIST with parameters from params.yaml

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow import keras

DATA_DIR, MODEL_DIR = "data/processed", "models"

def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["train"]
    tf.keras.utils.set_random_seed(p.get("seed", 42))
    os.makedirs(MODEL_DIR, exist_ok=True)

    train = np.load(f"{DATA_DIR}/train.npz")
    val = np.load(f"{DATA_DIR}/val.npz")

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(p["dense_units"], activation="relu"),
        keras.layers.Dropout(p["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train["x"], train["y"],
        validation_data=(val["x"], val["y"]),
        epochs=p["epochs"], batch_size=p["batch_size"], verbose=2,
    )
    model.save(f"{MODEL_DIR}/model.h5")
    pd.DataFrame(history.history).to_csv(f"{MODEL_DIR}/history.csv", index_label="epoch")
    print("Saved models/model.h5 and models/history.csv")

if __name__ == "__main__":
    main()