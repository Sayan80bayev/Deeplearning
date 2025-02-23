import numpy as np

Resources_Matrix = np.array([[200, 100], [150, 250], [300, 200]])
Allocation_Factors = np.array([[1.1, 0.9], [1.0, 1.2]])
Total_Resource_Distribution = np.dot(Resources_Matrix, Allocation_Factors)
print("Total Resource Distribution:")
print(Total_Resource_Distribution)

Shift_A_Production = np.array([50, 60, 55, 45, 70, 65, 80])
Shift_B_Production = np.array([40, 45, 50, 60, 55, 75, 85])
Total_Production = Shift_A_Production + Shift_B_Production
print("\nTotal Production:")
print(Total_Production)

Forecasted_Values = np.array([1.5, -0.2, 2.1, 0.8])
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

Sigmoid_Output = sigmoid(Forecasted_Values)
print("\nSigmoid Output:")
print(Sigmoid_Output)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

Sigmoid_Inputs = np.array([0.3, 1.2, -0.7, 1.8])
Sigmoid_Gradient = sigmoid_derivative(Sigmoid_Inputs)
print("\nSigmoid Gradient:")
print(Sigmoid_Gradient)


#neural network

X = np.array([[20, 3, 4],
              [15, 5, 3],
              [30, 2, 2],
              [25, 4, 1],
              [35, 2, 3]])

Y = np.array([[18],
              [20],
              [22],
              [25],
              [30]])

X = X / np.max(X, axis=0)
Y = Y / np.max(Y)

np.random.seed(42)
W1 = np.random.rand(3, 3)
b1 = np.random.rand(1, 3)
W2 = np.random.rand(3, 1)
b2 = np.random.rand(1, 1)

learning_rate = 0.1
epochs = 10000

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def backward_propagation(Z1, A1, Z2, A2, Y):
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, learning_rate):
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    return W1, b1, W2, b2

for epoch in range(epochs):

    Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)

    loss = np.mean(np.square(Y - A2)) / 2

    dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, Y)

    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, learning_rate)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")


print("Final Weights (W1):\n", W1)
print("Final Biases (b1):\n", b1)
print("Final Weights (W2):\n", W2)
print("Final Biases (b2):\n", b2)
print("Predictions:\n", A2)