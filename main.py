import pandas as pd
from sklearn.model_selection import train_test_split
from core.neural_network import NeuralNetwork
from core.activation_functions import Sigmoid

# Load data from CSV
data = pd.read_csv("dados_alarme.csv")

# Separate input (X) and output (y) data
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)  # Reshape to have a single column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define input, hidden, and output sizes, and other necessary configurations
input_size = X_train.shape[1]  # Number of features
hidden_size = 15  # Number of neurons in the hidden layer
output_size = 1  # Number of outputs
activation_function = Sigmoid()  # Assuming you have a Sigmoid class for the activation function
learning_rate = 0.1  # Learning rate

# Initialize the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size, activation_function, learning_rate)

# Train the neural network
epochs = 1000
nn.train(X_train, y_train, epochs)

# Predict outputs for training set
predictions_train, train_errors = nn.predict(X_train, y_train)

# Generate training report
nn.generate_report(predictions_train, y_train, train_errors)
