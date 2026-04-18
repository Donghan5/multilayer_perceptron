import pandas as pd
import numpy as np
from dataclasses import dataclass
from utils import standardize, sigmoid, sigmoid_prime, relu, relu_prime, softmax, cross_entropy, cross_entropy_prime, mse, mse_prime

@dataclass
class NetworkConfig:
	layers: list[int]
	activation: str
	loss: str
	output_activation: str
	weights_initializer: str = "heUniform"

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
		# Hidden layer activation
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
		
		if self.config.output_activation == "softmax":
			self.output_activation = softmax
		elif self.config.output_activation == "sigmoid":
			self.output_activation = sigmoid
		elif self.config.output_activation == "relu":
			self.output_activation = relu
		else:
			self.output_activation = self.activation

	def init_weights(self):
		for i in range(len(self.config.layers) - 1):
			input_dim = self.config.layers[i]
			output_dim = self.config.layers[i + 1]

			if self.config.weights_initializer == "heUniform":
				limit = np.sqrt(6 / input_dim)
				w = np.random.uniform(-limit, limit, size=(input_dim, output_dim))
			
			elif self.config.weights_initializer == "xavierUniform":
				limit = np.sqrt(6 / (input_dim + output_dim))
				w = np.random.uniform(-limit, limit, size=(input_dim, output_dim))
			else:
				raise ValueError(f"Unsupported initializer: {self.config.weights_initializer}")
			
			self.weights.append(w)
			self.biases.append(np.zeros(output_dim))
	
	def forward(self, x):
		self.activations = [x]
		self.zs = []
		
		curr_activation = x

		for i in range(len(self.weights) - 1):
			w = self.weights[i]
			b = self.biases[i]
			z = np.dot(curr_activation, w) + b
			self.zs.append(z)
			curr_activation = self.activation(z)
			self.activations.append(curr_activation)
		
		w = self.weights[-1]
		b = self.biases[-1]
		z = np.dot(curr_activation, w) + b
		self.zs.append(z)
		curr_activation = self.output_activation(z)
		self.activations.append(curr_activation)
			
		return curr_activation

	"""
		nabla_w = delta weight
		nabla_b = delta bias
	"""
	def backward(self, y_true):
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		
		y_pred = self.activations[-1]
		batch_size = y_true.shape[0]
		
		if self.config.loss == "cross_entropy":
			delta = (y_pred - y_true) / batch_size
		else:
			delta = self.loss_prime(y_true, y_pred) * self.activation_prime(y_pred) / batch_size
		
		nabla_w[-1] = np.dot(self.activations[-2].T, delta)
		nabla_b[-1] = np.sum(delta, axis=0)

		for l in range(2, len(self.config.layers)):
			z = self.zs[-l]

			delta = np.dot(delta, self.weights[-l + 1].T) * self.activation_prime(z)
			nabla_w[-l] = np.dot(self.activations[-l - 1].T, delta)
			nabla_b[-l] = np.sum(delta, axis=0)
		
		return nabla_w, nabla_b