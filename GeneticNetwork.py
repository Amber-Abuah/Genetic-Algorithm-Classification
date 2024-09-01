import numpy as np
from enum import Enum
import random
import copy

class ActivationFunction(Enum):
    NONE = 0,
    RELU = 1,
    SIGMOID = 2,
    TANH = 3
    
class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range (num_inputs)]
        self.bias = random.uniform(-1, 1)

    def forward_pass(self, inputs):
        return self.bias + sum([inputs[i] * self.weights[i] for i in range(len(inputs))])
    
    def get_weights(self):
        return self.weights
    
    def set_weights(self, weights):
        self.weights = weights

    def get_bias(self):
        return self.bias
    
    def set_bias(self, bias):
        self.bias = bias

    
class Layer:
    def __init__(self, num_neurons, num_inputs, activation_function : ActivationFunction = ActivationFunction.NONE):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
        self.activation_function = activation_function

    def forward_pass(self, inputs):
       return [self.apply_activation_function(n.forward_pass(inputs)) for n in self.neurons]
    
    def get_weights(self):
        return [n.get_weights() for n in self.neurons]
    
    def set_weights(self, weights):
        for i in range(len(self.neurons)):
            self.neurons[i].set_weights(weights[i])

    def get_biases(self):
        return [n.get_bias() for n in self.neurons]
    
    def set_biases(self, biases):
        for i in range(len(self.neurons)):
            self.neurons[i].set_bias(biases[i])

    def apply_activation_function(self, x):
        if self.activation_function == ActivationFunction.NONE:
            return x
        elif self.activation_function == ActivationFunction.RELU:
            return max(0, x)
        elif self.activation_function == ActivationFunction.SIGMOID:
            return 1/ (1 + np.exp(-x))
        else:
            return np.tanh(x)
    

class Network:
    def __init__(self, net_struct):
        self.net_struct = net_struct
        self.layers = []

        for i in range(len(net_struct[0])):
            if i == 0:
                self.layers.append(Layer(net_struct[0][i], net_struct[0][i]))
            else:
                self.layers.append(Layer(net_struct[0][i], net_struct[0][i - 1], net_struct[1][i]))

    def forward_pass(self, input):
        out = input
        for l in self.layers:
            out = l.forward_pass(out)
        return out
    
    def get_weights(self):
        return [l.get_weights() for l in self.layers]
    
    def set_weights(self, weights):
        for i in range(len(self.layers)):
            self.layers[i].set_weights(weights[i])

    def get_biases(self):
        return [l.get_biases() for l in self.layers]
    
    def set_biases(self, biases):
         for i in range(len(self.layers)):
            self.layers[i].set_biases(biases[i])

    def __str__(self) -> str:
        return str(self.get_weights()) + str(self.get_biases())

    @staticmethod
    def create_child_network(parent1_net, parent2_net):
        p2_rate = 0.45 # Rate of inheriting second parent genes 
        mutation_rate = 0.05 # Rate of randomisation per gene

        net1_weights = parent1_net.get_weights()
        net1_biases = parent1_net.get_biases()

        net2_weights = parent2_net.get_weights()
        net2_biases = parent2_net.get_biases()

        # Initially assign child's weight and biases parent 1's genes
        child_weights = copy.deepcopy(net1_weights)
        child_biases = copy.deepcopy(net1_biases)

        # Loop through all child weights and either assign parent 2's or random weight
        for i in range(len(child_weights)):
            for j in range(len(child_weights[i])):
                for k in range(len(child_weights[i][j])):
                    if random.uniform(0, 1) <= mutation_rate:
                        child_weights[i][j][k] = random.uniform(-1, 1)
                    elif random.uniform(0, 1) <= p2_rate:
                        child_weights[i][j][k] = net2_weights[i][j][k]

        # Loop through all child biases and either assign parent 2's or random bias
        for i in range(len(child_biases)):
            for j in range(len(child_biases[i])):
                if random.uniform(0, 1) <= mutation_rate:
                    child_biases[i][j] = random.uniform(-1, 1)
                elif random.uniform(0, 1) <= p2_rate:
                    child_biases[i][j] = net2_biases[i][j]

        # Create and return child network with assigned genes
        child_net = Network(parent1_net.net_struct)
        child_net.set_weights(child_weights)
        child_net.set_biases(child_biases)

        return child_net