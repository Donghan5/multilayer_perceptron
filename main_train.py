import pandas as pd
import numpy as np
from multilayer_perceptron import Mlp

def main():
	""" Main function to demonstrate MLP usage """
	try:
		df = pd.read_csv("data.csv", header=None)

		y_raw = df.iloc[:, 1].values
		X_raw = df.iloc[:, 2:].values

		y = np.zeros((len(y_raw), 2))

		# if label is 'M', then [0, 1], else [1, 0] --> one-hot encoding
		for i, label in enumerate(y_raw):
			if label == 'M':
				y[i] = [0, 1]
			else:
				y[i] = [1, 0]

		X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)

	except FileNotFoundError:
		print("data.csv not found. Please ensure the dataset is available.")
		return

	mlp = Mlp(
		hidden_layer_sizes=[16, 16],
		learning_rate=0.01,
		epochs=100,
		solver="adam"
	)

	print("Training the MLP model...")
	mlp.fit(X, y)
	print("Training completed.")

	mlp.save("model.pkl")

if __name__ == "__main__":
	main()