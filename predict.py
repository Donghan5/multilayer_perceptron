import numpy as np
import pandas as pd
import pickle	# To save and load the model (in serialized form)
import model
from utils import cross_entropy, one_hot_encode
import argparse

def predict(data_path, model_path):
	try:
		with open(model_path, 'rb') as f:
			model = pickle.load(f)
	except FileNotFoundError:
		print(f"Model file not found: {model_path}")
		return

	mean_train = model.mean_train
	std_train = model.std_train
	
	# test data
	try:
		df = pd.read_csv(data_path, header=None)

		y_raw = df.iloc[:, 1].values
		X = df.iloc[:, 2:].values

		# Standardize features
		X = (X - mean_train) / std_train

	except Exception as e:
		print(f"Error loading or processing data: {e}")
		return
	
	probabilities = model.predict(X)

	y_true_one_hot = one_hot_encode(y_raw)
	
	loss = cross_entropy(y_true_one_hot, probabilities)
	print(f"Loss: {loss:.4f}")

	predictions = np.argmax(probabilities, axis=1)

	labels = ["B", "M"]
	decoded_predictions = [labels[p] for p in predictions]

	return decoded_predictions

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", default="data.csv", help="Path to the input data CSV file")
	parser.add_argument("--model", default="model.pkl", help="Path to the trained model file")
	args = parser.parse_args()

	preds = predict(args.data, args.model)
	print(preds)