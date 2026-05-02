import numpy as np 
# ASCII values for '0' to '9' 
ascii_values = list(range(48, 58)) 
 
# Convert ASCII values to 6-bit binary 
inputs = [list(map(int, format(a, '06b'))) for a in ascii_values] 
 
# Target: Even = 1, Odd = 0 
targets = [1 if (a % 2 == 0) else 0 for a in ascii_values] 
 
# Initialize weights and bias 
weights = np.zeros(6) 
bias = 0 
learning_rate = 0.1 
 
# Step activation function 
def step(x): 
    return 1 if x >= 0 else 0 
 
# Training 
for epoch in range(25): 
    for x, target in zip(inputs, targets): 
        net_input = np.dot(weights, x) + bias 
        output = step(net_input) 
        error = target - output 
 
        weights += learning_rate * error * np.array(x) 
        bias += learning_rate * error 
 
print("Training Completed") 
print("Final Weights:", weights) 
print("Final Bias:", bias) 
 
# Testing 
print("\nASCII | Digit | Prediction (1=Even, 0=Odd)") 
print("-------------------------------------------") 
 
for a in ascii_values: 
    x = list(map(int, format(a, '06b'))) 
    prediction = step(np.dot(weights, x) + bias) 
    print(" ", a, " | ", chr(a), " | ", prediction)