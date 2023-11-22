import numpy as np

# inputs from the previous layer
inputs = [1.5, 1.5, 4.2]

# weights associated with every input
weights = [8.7, 6.5, 3.4]

# one bias per neuron
bias = 3

output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
np.array([5])
