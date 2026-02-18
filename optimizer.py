import numpy as np

"""
	SGD: Stochastic Gradient Descent
	Update rule: W = W - learning_rate * gradient

"""
class Sgd:
	def __init__(self, learning_rate: float):
		self.learning_rate = learning_rate

	"""
		SGD update
		W = W - learning_rate * gradient
	"""	
	def update(self, network, nabla_w, nabla_b):
		network.weights = [w - self.learning_rate * nw for w, nw in zip(network.weights, nabla_w)]
		network.biases = [b - self.learning_rate * nb for b, nb in zip(network.biases, nabla_b)]

"""
	ADAM: adapted moment estimation, combines advantages of Momentum and RMSprop techniques
"""
class Adam:
	def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.m_w = None	# 1st Momentum
		self.v_w = None	# 2nd RMSProp
		self.m_b = None	# 1st Momentum
		self.v_b = None	# 2nd RMSProp
		self.timestap = 0
	
	def update(self, network, nabla_w, nabla_b):
		if self.m_w is None:
			self.m_w = [np.zeros_like(w) for w in network.weights]
			self.v_w = [np.zeros_like(w) for w in network.weights]
			self.m_b = [np.zeros_like(b) for b in network.biases]
			self.v_b = [np.zeros_like(b) for b in network.biases]
		self.timestap += 1

		# Update weight and bias
		for i in range(len(network.weights)):
			self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * nabla_w[i]
			self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * nabla_w[i] ** 2
			self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * nabla_b[i]
			self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * nabla_b[i] ** 2
			
			m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.timestap)
			v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.timestap)
			m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.timestap)
			v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.timestap)
			network.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
			network.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
