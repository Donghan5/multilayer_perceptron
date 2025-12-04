import numpy as np
import pandas as pd
from network import Network, NetworkConfig
from optimizer import Sgd, Adam

class Model:
	def __init__(self, config: NetworkConfig) -> None:
		self.config = config
		self.network = Network(config)
	
	def fit(self, x_train, y_train, learning_rate, epochs, batch_size, optimization="sgd"):
		if optimization == "adam":
			optimizer = Adam(learning_rate)
		else:
			optimizer = Sgd(learning_rate)
		
		n_samples = len(x_train)

		for epoch in range(epochs):
			indices = np.arange(n_samples)
			np.random.shuffle(indices)

			if hasattr(x_train, 'iloc'):
				x_shuffled = x_train.iloc[indices].values
				y_shuffled = y_train.iloc[indices].values
			else:
				x_shuffled = x_train[indices]
				y_shuffled = y_train[indices]
		
		for i in range(0, n_samples, batch_size):
			x_batch = x_shuffled[i : i + batch_size]
			y_batch = y_shuffled[i : i + batch_size]

			self.network.forward(x_batch)
			nabla_w, nabla_b = self.network.backward(y_batch)
			optimizer.update(self.network, nabla_w, nabla_b)
			