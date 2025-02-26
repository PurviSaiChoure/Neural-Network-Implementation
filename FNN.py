import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self):
        self.weights_input_hidden = np.array([[20.0, 20.0], [-20.0, -20.0]])
        self.bias_hidden = np.array([-10.0, 30.0])
        
        self.weights_hidden_output = np.array([[20.0], [20.0]])
        self.bias_output = np.array([-30.0])
        
        self.hidden_output = None
        self.output = None
        
    def forward(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden.T) + self.bias_hidden
        self.hidden_output = sigmoid(hidden_input)
        
        output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(output_input)
        
        return self.output
    
    def backward(self, X, y_true, learning_rate=0.1):
        # Compute loss (Mean Squared Error)
        loss = np.mean((self.output - y_true) ** 2)
        print(f"Computed Loss: {loss:.6f}")
        
        d_loss = 2 * (self.output - y_true) / y_true.size  
        d_output = d_loss * sigmoid_derivative(self.output)
        
        d_weights_hidden_output = np.dot(self.hidden_output.T, d_output)
        d_bias_output = np.sum(d_output, axis=0)
        
        d_hidden = np.dot(d_output, self.weights_hidden_output.T) * sigmoid_derivative(self.hidden_output)
        d_weights_input_hidden = np.dot(X.T, d_hidden)
        d_bias_hidden = np.sum(d_hidden, axis=0)
        print("\nGradients for Hidden Layer Weights:")
        print(d_weights_input_hidden)
        print("\nGradients for Hidden Layer Biases:")
        print(d_bias_hidden)
        print("\nGradients for Output Layer Weights:")
        print(d_weights_hidden_output)
        print("\nGradient for Output Layer Bias:")
        print(d_bias_output)
    
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_true = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork()
y_pred = nn.forward(X)
binary_output = np.round(y_pred)

print("XOR Inputs:")
print(X)
print("\nHidden Layer Outputs:")
print(nn.hidden_output)
print("\nFinal Predictions (Before Thresholding):")
print(np.round(y_pred, 5))
print("\nFinal Predictions (Binary Output):")
print(binary_output)
print("\n--- Backpropagation Demonstration ---")
nn.backward(X, y_true)