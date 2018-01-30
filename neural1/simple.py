# Sigmoid function - "The Simplest Neural Network" section

import numpy as np

# exponential function can take inputs from an array
def sigmoid(x):
    # TODO: Implement sigmoid function
    val = 1 / ( 1 + np.exp( -x ))
    return val

inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
bias = -0.1

# TODO: Calculate the output
# y = f(h) = sigmoid( SUM( w_i * x_i + b ))
output = sigmoid( np.dot( inputs, weights ) + bias )

print('Output:')
print(output)