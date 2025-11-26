import pandas as pd
import numpy as np

from dataclasses import dataclass
from utils import standarize, sigmoid, sigmoid_prime, relu, relu_prime, softmax

@dataclass
class NetworkConfig:
	layers: list[int]
	activation: str
	loss: str

class Network:
	def __init__(self, config: NetworkConfig) -> None:
		self.config = config
		self.network = []
		self.weights = []
		self.biases = []
		self.gw_history = []
		self.gb_history = []
		self.activations = []
		self.zs = []
		self.build_network()
		self.init_weights()
	
	def build_network(self):
		"""
			Decide activation function and loss function
		"""
		if self.config.activation == "sigmoid":
			self.activation = sigmoid
			self.activation_prime = sigmoid_prime
		elif self.config.activation == "relu":
			self.activation = relu
			self.activation_prime = relu_prime
		else:
			self.activation = self.config.activation
			self.activation_prime = self.config.activation_prime
		if self.config.loss == "mse":
			self.loss = mse
			self.loss_prime = mse_prime
		elif self.config.loss == "cross_entropy":
			self.loss = cross_entropy
			self.loss_prime = cross_entropy_prime
		else:
			self.loss = self.config.loss
			self.loss_prime = self.config.loss_prime

	def init_weights(self):
		for i in range(len(self.config.layers) - 1):
			input_dim = self.config.layers[i]
			output_dim = self.config.layers[i + 1]
			
			scale = 0.1 
			self.weights.append(np.random.randn(input_dim, output_dim) * scale)
			self.biases.append(np.zeros(output_dim))
	
	def forward(self, x):
		self.activations = [x]
		self.zs = []
		
		curr_activation = x

		for w, b in zip(self.weights, self.biases):
			z = np.dot(curr_activation, w) + b
			self.zs.append(z)
			curr_activation = self.activation(z)
			self.activations.append(curr_activation)
			
		return curr_activation

	def backward(self, loss_gradient):
		
		
