import numpy as np 
import matplotlib.pyplot as plt 
 
x = np.array([[0,0], 
              [0,1], 
              [1,0], 
              [1,1]]) 
 
y = np.array([-1,-1,-1,1]) 
 
w1 = 0 
w2 = 0 
b = 0 
alpha = 1 
 
for epochs in range(10): 
    for i in range(len(x)): 
 
        yin = w1*x[i][0] + w2*x[i][1] + b 
 
        if yin >= 0: 
                 y_pred = 1 
        else: 
            y_pred = -1 
 
        if y_pred != y[i]: 
            w1 += alpha * y[i] * x[i][0] 
            w2 += alpha * y[i] * x[i][1] 
            b += alpha * y[i] 
 
print("Final Weights:", w1, w2) 
print("Final Bias:", b) 
 
for i in range(len(x)): 
    yin = w1*x[i][0] + w2*x[i][1] + b 
    if yin >= 0: 
        y_pred = 1 
    else: 
        y_pred = -1 
    print(x[i], "->", y_pred) 
 
## graphical representation 
plt.figure() 
 
for i in range(len(x)): 
    if y[i] == 1: 
        plt.scatter(x[i][0], x[i][1], color='blue') 
    else: 
        plt.scatter(x[i][0], x[i][1], color='red') 
 
if w2 != 0: 
    x1_1 = -0.5 
    x2_1 = -(w1*x1_1 + b)/w2
    x1_2 = 1.5 
    x2_2 = -(w1*x1_2 + b)/w2 
 
    plt.plot([x1_1, x1_2], [x2_1, x2_2]) 
else: 
    plt.axvline(x=-b/w1) 
 
plt.xlim(-0.5, 1.5) 
plt.ylim(-0.5, 1.5) 
plt.xlabel("x1") 
plt.ylabel("x2") 
plt.title("Perceptron AND Gate") 
plt.grid() 
plt.show()