# evaluates trained model on test data, saves a confusion matrix plot to 'reports/', and logs metrics to 'metrics.json'import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def main():
    os.makedirs("reports", exist_ok=True)
    model = keras.models.load_model("models/model.h5")
    test = np.load("data/processed/test.npz")

    loss, acc = model.evaluate(test["x"], test["y"], verbose=0)
    y_pred = np.argmax(model.predict(test["x"], verbose=0), axis=1)

    cm = confusion_matrix(test["y"], y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=120)

    with open("metrics.json", "w") as f:
        json.dump({"test_loss": round(float(loss), 4),
                   "test_accuracy": round(float(acc), 4)}, f, indent=2)
    print(f"test_loss={loss:.4f}  test_accuracy={acc:.4f}")

if __name__ == "__main__":
    main()