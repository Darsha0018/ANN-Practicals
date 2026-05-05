import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# XOR input and output
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize weights
np.random.seed(1)
W1 = np.random.rand(2, 3)   # input → hidden
W2 = np.random.rand(3, 1)   # hidden → output

# Bias (new variables)
b1 = np.random.rand(1, 3)
b2 = np.random.rand(1, 1)

# Training parameters
lr = 0.5
epochs = 10000

# Training loop
for epoch in range(epochs):

    # ---- Forward Propagation ----
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # ---- Error ----
    error = y - final_output

    # ---- Backpropagation ----
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # ---- Update weights ----
    W2 += hidden_output.T.dot(d_output) * lr
    W1 += X.T.dot(d_hidden) * lr

    # ---- Update bias ----
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    # ---- Print Loss ----
    if epoch % 2000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ---- Final Output ----
print("\nFinal Output (Raw):")
print(final_output)

# ---- Binary Output ----
print("\nBinary Output:")
print(np.round(final_output))