import numpy as np
import pandas as pd
import pickle	# To save and load the model (in serialized form)
from multilayer_perceptron import MultilayerPerceptron
from utils import cross_entropy
import argparse

def predict(data_path, model_path):
	"""
		Load the trained model and make predictions on the input data
	"""
	try:
		with open(model_path, 'rb') as f:
			model = pickle.load(f)
	except FileNotFoundError:
		print(f"Model file not found: {model_path}")
		return
	
	try:
		df = pd.read_csv(data_path, header=None)

		y_raw = df.iloc[:, 1].values

		# Encode labels: 'M' -> 1, 'B' -> 0
		y_true = np.array([1 if label == 'M' else 0 for label in y_raw])

		X = df.iloc[:, 2:].values

		# Standardize features
		X = (X - X.mean(axis=0)) / X.std(axis=0)
	except Exception as e:
		print(f"Error loading or processing data: {e}")
		return
	
	probabilities = model.predict(X)

	y_prob = probabilities[:, 1]

	loss = cross_entropy(y_true, y_prob)

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