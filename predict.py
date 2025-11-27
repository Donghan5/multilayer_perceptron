import numpy as np
import pandas as pd
from network import Network, NetworkConfig

def predict(data_path, model_path):
	data = pd.read_csv(data_path)

	config = NetworkConfig(layers=[...], activation="relu", loss="cross_entropy")
	model = Network(config)
	
	weights, biases = load_weights(model_path)