# Normalizes Fashion-MNIST images to [0, 1], stratifies train into train/val splits, and saves to 'data/processed'
# stratify keeps class proportions equal in train and validation
import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW_DIR, OUT_DIR = "data/raw", "data/processed"

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]
    os.makedirs(OUT_DIR, exist_ok=True)
    train = np.load(f"{RAW_DIR}/train.npz")
    test = np.load(f"{RAW_DIR}/test.npz")

    # Normalization: uint8 [0,255] -> float32 [0,1]
    x_train = train["x"].astype("float32") / 255.0
    x_test = test["x"].astype("float32") / 255.0

    x_train = (x_train - 0.5) / 0.5
    x_test = (x_test - 0.5) / 0.5

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, train["y"],
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=train["y"],
    )
    np.savez_compressed(f"{OUT_DIR}/train.npz", x=x_tr, y=y_tr)
    np.savez_compressed(f"{OUT_DIR}/val.npz", x=x_val, y=y_val)
    np.savez_compressed(f"{OUT_DIR}/test.npz", x=x_test, y=test["y"])
    print(f"train {x_tr.shape} | val {x_val.shape} | test {x_test.shape}")

if __name__ == "__main__":
    main()