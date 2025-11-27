import numpy as np
import pandas as pd

from dataclasses import dataclass
from network import Network, NetworkConfig
from optimizer import Sgd, Adam

@dataclass
class TrainConfig:
	"""
		Hyperparameters for training the network
	"""
	model_config: NetworkConfig
	epochs: int
	batch_size: int
	learning_rate: float
	optimization: str
	
class Train:
	def __init__(self, config: TrainConfig) -> None:
		self.config = config
		self.network = Network(config.model_config)
		self.optimizer = self._get_optimizer()
	
	def _get_optimizer(self) -> None:
		if self.config.optimization == "adam":
			return Adam(self.config.learning_rate)
		else:
			return Sgd(self.config.learning_rate)
		
	def train(self, x_train, y_train):
		n_samples = len(x_train)
		
		for epoch in range(self.config.epochs):
			indices = np.arange(n_samples)
			np.random.shuffle(indices)
			
			x_shuffled = x_train.iloc[indices].values if hasattr(x_train, 'iloc') else x_train[indices]
			y_shuffled = y_train.iloc[indices].values if hasattr(y_train, 'iloc') else y_train[indices] 
			
			for i in range(0, n_samples, self.config.batch_size):
				x_batch = x_shuffled[i : i + self.config.batch_size]
				y_batch = y_shuffled[i : i + self.config.batch_size]
				
				self.network.forward(x_batch)

				nable_w, nabla_b = self.network.backward(y_batch)
				self.optimizer.update(self.network, nabla_w, nabla_b)
