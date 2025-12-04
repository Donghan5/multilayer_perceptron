import numpy as np
import pandas as pd
from model import Mlp

def predict(data_path, model_path):
	data = pd.read_csv(data_path)

	model = Mlp(
		hiddden_layer_sizes=m_data['hidden_layer_sizes'],
		output_layer_size=m_data['output_layer_size'],
		activation=m_data['activation'],
		output_activation=m_data['output_activation'],
		loss=m_data['loss']
	)

	try:
		x = pd.read_csv(data_path, header=None)
	except:
		print(f"{YELLOW}Error loading data: {e}{END}")
		exit(1)
	
	return model.network.forward(x)