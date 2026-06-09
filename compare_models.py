# TO-DO 
"""
Task 1:
Create compare_models.py

Required behavior:
- load train.csv and validation.csv
- validate_dataset()
- extract X/y using same schema as main.py
- one_hot_encode labels
- train 2 configs: Adam vs SGD
- collect histories
- plot val_loss of both models on one graph
- plot val_accuracy of both models on one graph
- save comparison curves
- save histories
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import Model
from utils import one_hot_encode, validate_dataset
import json

def import_csvs():
    """
    Loads the training and validation datasets from CSV files
    """
    try:
        df_train = pd.read_csv("train.csv", header=None)
        df_val = pd.read_csv("validation.csv", header=None)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Run 'python split.py' to split the dataset into train and validation sets.")
        return None, None

    return df_train, df_val

def validate_and_extract(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validates the datasets and extracts features and labels
    """
    validate_dataset(df_train, name="Training dataset")
    validate_dataset(df_val, name="Validation dataset")

    X_train = df_train.iloc[:, 2:].values.astype(float)
    y_train = df_train.iloc[:, 1].values
    X_val = df_val.iloc[:, 2:].values.astype(float)
    y_val = df_val.iloc[:, 1].values

    y_train = one_hot_encode(y_train)
    y_val = one_hot_encode(y_val)

    return X_train, y_train, X_val, y_val

def train_model_with_adam(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
) -> dict[str, list]:
    """
    Trains the MLP model using the Adam optimizer and returns the training history
    """
    mlp_adam = Model(
        hidden_layer_sizes=[24, 24, 24],
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
        solver="adam",
        early_stopping_rounds=5,
        min_delta=0.001
    )

    history_adam = mlp_adam.fit(X_train, y_train, x_val=X_val, y_val=y_val)

    return history_adam

def train_model_with_sgd(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
) -> dict[str, list]:
    """
    Trains the MLP model using the SGD optimizer and returns the training history
    """
    mlp_sgd = Model(
        hidden_layer_sizes=[24, 24, 24],
        learning_rate=0.001,
        epochs=200,
        batch_size=32,
        solver="sgd",
        early_stopping_rounds=5,
        min_delta=0.001
    )

    history_sgd = mlp_sgd.fit(X_train, y_train, x_val=X_val, y_val=y_val)

    return history_sgd

def plot_comparison_curves(history_adam: dict[str, list], history_sgd: dict[str, list]) -> None:
    """
    Plots the validation loss and accuracy curves for both Adam and SGD optimizers
    """

    # Create epoch ranges for plotting
    adam_loss_epochs = range(1, len(history_adam["val_loss"]) + 1)
    sgd_loss_epochs = range(1, len(history_sgd["val_loss"]) + 1)

    adam_acc_epochs = range(1, len(history_adam["val_accuracy"]) + 1)
    sgd_acc_epochs = range(1, len(history_sgd["val_accuracy"]) + 1)
                       
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(adam_loss_epochs, history_adam['val_loss'], label='Adam Validation Loss')
    plt.plot(sgd_loss_epochs, history_sgd['val_loss'], label='SGD Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(adam_acc_epochs, history_adam['val_accuracy'], label='Adam Validation Accuracy')
    plt.plot(sgd_acc_epochs, history_sgd['val_accuracy'], label='SGD Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('optimizer_comparison.png')
    print("Comparison curves saved to optimizer_comparison.png")

    plt.close()

def make_json_serializable(history: dict[str, list]) -> dict[str, list]:
    """
    Converts the training history to a JSON-serializable format
    """
    return {
        key: [float(value) for value in values] 
        for key, values in history.items()
    }

def save_histories(history_adam: dict[str, list], history_sgd: dict[str, list]) -> None:
    """
    Saves the training histories for both Adam and SGD optimizers to json files
    """
    with open("optimizer_histories.json", "w") as f:
        json.dump(
            {
                "adam": make_json_serializable(history_adam), 
                "sgd": make_json_serializable(history_sgd)
            }, 
            f,
            indent=2
        )
    print("Optimizer histories saved to optimizer_histories.json")


def main():
    df_train, df_val = import_csvs()
    if df_train is None or df_val is None:
        raise FileNotFoundError("Required CSV files not found. Please run 'python split.py' to create train.csv and validation.csv.")

    X_train, y_train, X_val, y_val = validate_and_extract(df_train, df_val)

    np.random.seed(42)  # Set a fixed random seed for reproducibility
    history_adam = train_model_with_adam(X_train, y_train, X_val, y_val)

    np.random.seed(42)  # Reset the random seed to ensure fair comparison
    history_sgd = train_model_with_sgd(X_train, y_train, X_val, y_val)

    save_histories(history_adam, history_sgd)

    plot_comparison_curves(history_adam, history_sgd)

if __name__ == "__main__":
    main()