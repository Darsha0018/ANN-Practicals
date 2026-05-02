# McCulloch-Pitts Neuron for ANDNOT Function 
# Excitatory weight 
w = 1 
 
# Threshold 
theta = 1 
 
# Inputs (A = excitatory, B = inhibitory) 
inputs = [ 
    (0, 0), 
    (0, 1), 
    (1, 0), 
    (1, 1) 
] 
print("A B | Output") 
print("-------------") 
 
for A, B in inputs:
    # Check inhibitory input first
    if B == 1:
        output = 0
    else:
        net_input = w * A

        if net_input >= theta:
            output = 1
        else:
            output = 0
    print(A, B, "|", output)