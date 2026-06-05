import numpy as np
import pandas as pd
import argparse
from model import Model
from utils import binary_cross_entropy, classification_metrics, one_hot_encode, validate_dataset

def predict(data_path, model_path):
    try:
        mlp = Model.load(model_path)
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        df = pd.read_csv(data_path, header=None)
        validate_dataset(df, name="Input dataset")

        X = df.iloc[:, 2:].values.astype(float)
        y_raw = df.iloc[:, 1].values

    except Exception as e:
        print(f"Error loading or processing data: {e}")
        return


    probabilities = mlp.predict(X)
    predictions = np.argmax(probabilities, axis=1)

    y_true_one_hot = one_hot_encode(y_raw)
    y_true = np.argmax(y_true_one_hot, axis=1)

    p_malignant = probabilities[:, 1]
    loss = binary_cross_entropy(y_true, p_malignant)
    accuracy = np.mean(predictions == y_true)

    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    metrics = classification_metrics(y_true, predictions, positive_class=1)
    print(f"Precision(M): {metrics['precision']:.4f}")
    print(f"Recall(M): {metrics['recall']:.4f}")
    print(f"F1(M): {metrics['f1']:.4f}")
    print(f"Specificity(B): {metrics['specificity']:.4f}")
    print(
        "Confusion Matrix:\n"
        f"TP: {metrics['tp']}  FP: {metrics['fp']}\n"
        f"FN: {metrics['fn']}  TN: {metrics['tn']}"
    )

    labels = ["B", "M"]
    decoded_predictions = [labels[p] for p in predictions]

    return decoded_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="validation.csv", help="Path to the input validation CSV file")
    parser.add_argument("--model", default="model.npz", help="Path to the trained model file")
    args = parser.parse_args()

    results = predict(args.data, args.model)
    if results is not None:
        print(results)