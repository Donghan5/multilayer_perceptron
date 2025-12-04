import numpy as np
import pandas as pd
import pickle	# To save and load the model (in serialized form)
from multilayer_perceptron import Mlp

def predict(data_path, model_path):
	try:
		with open(model_path, 'rb') as f:
			model = pickle.load(f)
	except FileNotFoundError:
		print(f"Model file not found: {model_path}")
		return
	
	try:
		df = pd.read_csv(data_path, header=None)

		X = df.values

		# Standardize features
		X = (X - X.mean(axis=0)) / X.std(axis=0)
	except Exception as e:
		print(f"Error loading or processing data: {e}")
		return
	
	probabilities = model.model.forward(X)

	predictions = np.argmax(probabilities, axis=1)

	labels = ["B", "M"]
	decoded_predictions = [labels[p] for p in predictions]

	return decoded_predictions

if __name__ == "__main__":
	preds = predict("data.csv", "model.pkl")
	print(preds)