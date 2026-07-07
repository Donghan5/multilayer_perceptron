import argparse
from predict import predict

"""
Wrapper script for evaluating a trained MLP model on validation data. (Predict)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained MLP model.")
    parser.add_argument("--data", default="validation.csv", help="Path to the validation data CSV file.")
    parser.add_argument("--model", default="model.npz", help="Path to the trained model file.")
    args = parser.parse_args()

    result = predict(data_path=args.data, model_path=args.model)
    if result is not None:
        print(result)