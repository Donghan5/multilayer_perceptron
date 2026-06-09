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
from optimizer import Adam, Sgd
from utils import one_hot_encode, validate_dataset

def import_csvs() -> tuple[pd.DataFrame, pd.DataFrame]:
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
        epochs=50,
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
        learning_rate=0.01,
        epochs=50,
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
    Showing the learning curves side by side for easy comparison
    """

    epochs_adam = range(1, len(history_adam['loss']) + 1)
    epochs_sgd = range(1, len(history_sgd['loss']) + 1)
                       
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_adam, history_adam['val_loss'], label='Adam Validation Loss')
    plt.plot(epochs_sgd, history_sgd['val_loss'], label='SGD Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_adam, history_adam['val_accuracy'], label='Adam Validation Accuracy')
    plt.plot(epochs_sgd, history_sgd['val_accuracy'], label='SGD Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_adam, history_adam['learning_rate'], label='Adam Learning Rate')
    plt.plot(epochs_sgd, history_sgd['learning_rate'], label='SGD Learning Rate')
    plt.title('Learning Rate Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('optimizer_comparison.png')
    print("Comparison curves saved to optimizer_comparison.png")

def main():
    df_train, df_val = import_csvs()
    if df_train is None or df_val is None:
        raise FileNotFoundError("Required CSV files not found. Please run 'python split.py' to create train.csv and validation.csv.")
        return

    X_train, y_train, X_val, y_val = validate_and_extract(df_train, df_val)

    np.random.seed(42)  # Set a fixed random seed for reproducibility
    history_adam = train_model_with_adam(X_train, y_train, X_val, y_val)
    history_sgd = train_model_with_sgd(X_train, y_train, X_val, y_val)

    plot_comparison_curves(history_adam, history_sgd)

if __name__ == "__main__":
    main()