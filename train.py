import numpy as np
import pandas as pd

from dataclasses import dataclass
from network import Network, NetworkConfig


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
	
class TrainNetwork:
	def __init__(self, config: TrainConfig) -> None:
		self.config = config
		self.network = Network(config.model_config)