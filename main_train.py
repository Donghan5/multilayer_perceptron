import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from multilayer_perceptron import MultilayerPerceptron

def get_args():
	parser = argparse.ArgumentParser(description="Train the MLP model.")
	parser.add_argument("--data", type=str, default="data.csv", help="Path to the training data CSV file.")
	parser.add_argument("--layers", type=int, nargs='+', default=[24, 24, 24], help="Sizes of hidden layers.")
	parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
	parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
	parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
	parser.add_argument("--split", type=float, default=0.2, help="Train/validation split ratio.")
	return parser.parse_args()


def one_hot_encode(y):
	encoded = np.zeros((len(y), 2))
	for i, label in enumerate(y):
		if label == 'M':
			encoded[i] = [0, 1]
		else:
			encoded[i] = [1, 0]
	return encoded

def plot_learning_curve(history):
	epochs = range(1, len(history['loss']) + 1)

	plt.figure(figsize=(12, 5))

	plt.subplot(1, 2, 1)
	plt.plot(epochs, history['loss'], label='Training Loss')
	if 'val_loss' in history and len(history['val_loss']) > 0:
		plt.plot(epochs, history['val_loss'], label='Validation Loss')
	plt.title('Loss Curve')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.grid(True)
	
	plt.subplot(1, 2, 2)
	plt.plot(epochs, history['accuracy'], label='Training Accuracy')
	if 'val_accuracy' in history and len(history['val_accuracy']) > 0:
		plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
	plt.title('Accuracy Curve')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.grid(True)

	plt.tight_layout()
	plt.savefig('learning_curve.png')
	print("Learning curve saved to learning_curve.png")


def main():
	""" Main function to demonstrate MLP usage """
	args = get_args()
	try:
		df = pd.read_csv(args.data, header=None)

		y_raw = df.iloc[:, 1].values
		X_raw = df.iloc[:, 2:].values

		y = one_hot_encode(y_raw)

		X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)

	except FileNotFoundError:
		print("data.csv not found. Please ensure the dataset is available.")
		return
	
	indices = np.arange(len(X))
	np.random.shuffle(indices)

	split_idx = int(len(X) * (1 - args.split))

	train_idx, val_idx = indices[:split_idx], indices[split_idx:]
	X_train, X_val = X[train_idx], X[val_idx]
	y_train, y_val = y[train_idx], y[val_idx]

	print(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples.")

	mlp = MultilayerPerceptron(
		hidden_layer_sizes=args.layers,
		learning_rate=args.learning_rate,
		epochs=args.epochs,
		batch_size=args.batch_size,
		solver="adam",
		output_activation="softmax",
		loss="cross_entropy"
	)

	print("Training the MLP model...")
	history = mlp.fit(X_train, y_train, X_val, y_val)
	print("Training completed.")

	mlp.save("model.pkl")

	plot_learning_curve(history)

if __name__ == "__main__":
	main()