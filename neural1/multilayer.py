#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 13:14:05 2017
\File    multilayer.py
@author: robert
"""

# TODO:
"""    
    Calculate the input to the hidden layer.
    Calculate the hidden layer output.
    Calculate the input to the output layer.
    Calculate the output of the network.
"""

import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

# Network size
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)
# Make some fake data
X = np.random.randn(4)

weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))


# TODO: Make a forward pass through the network
#print( "X size = ", X.shape )
#print( "weights_input_to_hidden size = ", weights_input_to_hidden.shape )
hidden_layer_in = np.matmul( X, weights_input_to_hidden )
hidden_layer_out = sigmoid( hidden_layer_in )

print('Hidden-layer Output:')
print(hidden_layer_out)

#print( "hidden_layer_out size = ", hidden_layer_out.shape )
#print( "weights_hidden_to_output size = ", weights_hidden_to_output.shape )
output_layer_in = np.matmul( hidden_layer_out, weights_hidden_to_output )
output_layer_out = sigmoid( output_layer_in )

print('Output-layer Output:')
print(output_layer_out)