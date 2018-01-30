#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:55:22 2017

@author: robert
"""

"""
TODO:
    
 * Calculate the network's output error.
 * Calculate the output layer's error term.
 * Use backpropagation to calculate the hidden layer's error term.
 * Calculate the change in weights (the delta weights) that result from 
     propagating the errors back through the network.
"""

import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))

# Inputs
x = np.array([0.5, 0.1, -0.2])
#print( "inputs: ", x )
#print( "inputs as column vector: ", x[:,None] )

# y
target = 0.6
# eta
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
print( "hidden_layer_input ", hidden_layer_input)

hidden_layer_output = sigmoid(hidden_layer_input)
print( "hidden_layer_output ", hidden_layer_output)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)
print( "output ", output)

## Backwards pass
## TODO: Calculate output error:  y - y_hat
error = target - output
print( "error ", error)

# TODO: Calculate error term for output layer
# delta_o = ( y - y_hat ) * f'(W dot a) = ( y - y_hat ) * f(h) * (1-f(h))
output_error_term = error * output * ( 1 - output )
print( "output_error_term ", output_error_term)

# TODO: Calculate error term for hidden layer
# delta_h = W * delta_o * f'(h) = W * delta_o * f(h) * (1-f(h))
hidden_error_term = weights_hidden_output * output_error_term * hidden_layer_output * ( 1 - hidden_layer_output)
# hidden_error_term = np.dot(output_error_term, weights_hidden_output) * \
#                    hidden_layer_output * (1 - hidden_layer_output)
print( "hidden_error_term ", hidden_error_term)

# TODO: Calculate change in weights for hidden layer to output layer
# del_w_h_o = eta * del_o * f(h), where f(h) is the hidden layer's output
delta_w_h_o = learnrate * output_error_term * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
# del_w_i = eta * del_h * x_i
delta_w_i_h = learnrate * hidden_error_term * x[:,None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
