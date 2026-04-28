import pandas as pd
import numpy as np
from dataclasses import dataclass
from utils import sigmoid, sigmoid_prime, relu, relu_prime, softmax, cross_entropy

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
		self.weights = []
		self.biases = []
		
		# Cache for backpropagation (Filled in Forward pass)
		self.zs = []    # pre-activation of each layers (z = W*a_prev + b)
		self.activations = []   # post-activation of each layers (a = g(z)). [0] is input x
		self.build_network()
		self.init_weights()
	
	def build_network(self):
		"""
			Decide activation function and loss function.
			Only supports: sigmoid/relu hidden activation + softmax output + cross_entropy loss.
		"""
		# Hidden layer activation
		if self.config.activation == "sigmoid":
			self.activation = sigmoid
			self.activation_prime = sigmoid_prime
		elif self.config.activation == "relu":
			self.activation = relu
			self.activation_prime = relu_prime
		else:
			raise ValueError(f"Unsupported activation: {self.config.activation}")
		
		# Output activation: softmax only
		if self.config.output_activation == "softmax":
			self.output_activation = softmax
		else:
			raise ValueError(f"Unsupported output_activation: {self.config.output_activation}. Only 'softmax' is supported.")
		
		# Loss: cross_entropy only
		if self.config.loss == "cross_entropy":
			self.loss = cross_entropy
		else:
			raise ValueError(f"Unsupported loss: {self.config.loss}. Only 'cross_entropy' is supported.")

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
		"""
			Forward pass. Cache the value in self.zs and self.activations.
			We are going to use these values in backward chain rule calculation.
		"""
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

	def backward(self, y_true):
		"""
			nabla_w = delta weight
			nabla_b = delta bias
		"""
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		
		y_pred = self.activations[-1]
		batch_size = y_true.shape[0]
		
		delta = (y_pred - y_true) / batch_size
		
		nabla_w[-1] = np.dot(self.activations[-2].T, delta)
		nabla_b[-1] = np.sum(delta, axis=0)

		for l in range(2, len(self.config.layers)):
			z = self.zs[-l]
			delta = np.dot(delta, self.weights[-l + 1].T) * self.activation_prime(z)
			nabla_w[-l] = np.dot(self.activations[-l - 1].T, delta)
			nabla_b[-l] = np.sum(delta, axis=0)
		
		return nabla_w, nabla_b