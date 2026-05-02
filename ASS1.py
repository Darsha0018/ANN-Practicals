import numpy as np 
import matplotlib.pyplot as plt 
 
class Neuron: 
    def __init__(self, weights, bias, activation): 
        self.weights = np.array(weights) 
        self.bias = bias 
        self.activation = activation 
 
    def forward(self, inputs): 
        inputs = np.array(inputs) 
        z = np.dot(inputs, self.weights) + self.bias 
        return self.activation(z) 
 
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 
 
def relu(x): 
    return np.maximum(0, x) 
 
def tanh(x): 
     return np.tanh(x) 
 
# create neurons using the same class : # Same weights and bias for fair comparison 

weights = [1.0] 
bias = 0.0 
 
sigmoid_neuron = Neuron(weights, bias, sigmoid) 
relu_neuron = Neuron(weights, bias, relu) 
tanh_neuron = Neuron(weights, bias, tanh) 
 
# plotting the activation function : 
 
import matplotlib.pyplot as plt 
 
# Input range 
x = np.array([ 
    -10, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 10  
]) 
 
# Forward pass through neurons 
y_sigmoid = [sigmoid_neuron.forward([i]) for i in x] 
y_relu = [relu_neuron.forward([i]) for i in x] 
y_tanh = [tanh_neuron.forward([i]) for i in x] 
 
# Plotting 
# 1. Sigmoid 
plt.figure() 
plt.plot(x, y_sigmoid, marker='o') 
plt.title("Sigmoid Function") 
plt.xlabel("Input") 
plt.ylabel("Output") 
plt.grid(True) 
plt.show() 
 
# 2. Tanh 
plt.figure() 
plt.plot(x, y_tanh, marker='o', color='green') 
plt.title("Tanh Function") 
plt.xlabel("Input") 
plt.ylabel("Output") 
plt.grid(True) 
plt.show() 
 
# 3. ReLU 
plt.figure() 
plt.plot(x, y_relu, marker='o', color='red') 
plt.title("ReLU Function") 
plt.xlabel("Input") 
plt.ylabel("Output") 
plt.grid(True) 
plt.show()