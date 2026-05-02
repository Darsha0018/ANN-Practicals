import numpy as np 
 
# Input Dataset (XOR)  
X = np.array([ 
    [0, 0], 
    [0, 1], 
    [1, 0], 
    [1, 1] 
]) 
 
Y = np.array([ 
    [0], 
    [1], 
    [1], 
    [0] 
]) 
W1 = np.array([[0.5, -0.5, 0.3, -0.3], 
               [0.4,  0.2, -0.4, 0.1]]) 
 
b1 = np.array([[0.0, 0.0, 0.0, 0.0]])
W2 = np.array([[0.2], 
               [-0.3], 
               [0.4], 
               [0.1]]) 
 
b2 = np.array([[0.0]]) 
 
# Activation Functions  
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 
 
def sigmoid_derivative(x): 
    return x * (1 - x) 
 
# Forward Propagation Function 
def forward_propagation(X, W1, b1, W2, b2): 
    z1 = np.dot(X, W1) + b1 
    a1 = sigmoid(z1) 
 
    z2 = np.dot(a1, W2) + b2 
    y_hat = sigmoid(z2) 
 
    return z1, a1, z2, y_hat 
 
# Backward Propagation Function 
def backward_propagation(X, y, z1, a1, z2, y_hat, W2): 
    error = y_hat - y 
    d_output = error * sigmoid_derivative(y_hat) 
 
    dW2 = np.dot(a1.T, d_output) 
    db2 = np.sum(d_output, axis=0, keepdims=True) 
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(a1) 
 
    dW1 = np.dot(X.T, d_hidden) 
    db1 = np.sum(d_hidden, axis=0, keepdims=True) 
 
    return dW1, db1, dW2, db2 
 
# Training Step Function 
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate): 
    W1 -= learning_rate * dW1 
    b1 -= learning_rate * db1 
 
    W2 -= learning_rate * dW2 
    b2 -= learning_rate * db2 
 
    return W1, b1, W2, b2 
 
learning_rate = 0.1 
 
# Forward 
z1, a1, z2, y_hat = forward_propagation(X, W1, b1, W2, b2) 
print("Forward Output (y_hat):") 
print(np.round(y_hat, 4)) 
 
# Backward 
dW1, db1, dW2, db2 = backward_propagation(X, Y, z1, a1, z2, y_hat, W2) 
 
# Update 
W1, b1, W2, b2 = update_parameters( 
    W1, b1, W2, b2, 
    dW1, db1, dW2, db2, 
    learning_rate) 
 
print("\nUpdated W1:") 
print(W1) 
 
print("\nUpdated W2:") 
print(W2) 