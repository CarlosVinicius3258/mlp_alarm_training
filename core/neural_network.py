# neural_network.py

from typing import Tuple
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from matplotlib.backends.backend_pdf import PdfPages


from core.activation_functions import ActivationFunction

class NeuralNetwork:
    """Implementação de uma rede neural feedforward com uma camada oculta."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, activation_function, learning_rate: float = 0.1):
        """
        Inicializa a rede neural com os parâmetros fornecidos.

        Args:
            input_size (int): Tamanho da camada de entrada.
            hidden_size (int): Tamanho da camada oculta.
            output_size (int): Tamanho da camada de saída.
            activation_function (ActivationFunction): Função de ativação a ser usada nos neurônios.
            learning_rate (float, optional): Taxa de aprendizado da rede. Defaults to 0.1.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.train_errors = []
        self._initialize_weights()

    def _initialize_weights(self):
        """Inicializa os pesos e vieses da rede neural."""
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_input_hidden = np.random.rand(self.hidden_size)
        self.bias_hidden_output = np.random.rand(self.output_size)

    def feedforward(self, inputs):
        """Executa o passo forward da rede neural."""
        net_hidden = np.dot(inputs, self.weights_input_hidden) + self.bias_input_hidden
        activation_hidden = self.activation_function(net_hidden)
        net_output = np.dot(activation_hidden, self.weights_hidden_output) + self.bias_hidden_output
        output = self.activation_function(net_output)
        return output

    def backpropagation(self, inputs, targets):
        """Executa o algoritmo de retropropagação para atualizar os pesos da rede neural."""
        # Forward pass
        net_hidden = np.dot(inputs, self.weights_input_hidden) + self.bias_input_hidden
        activation_hidden = self.activation_function(net_hidden)
        net_output = np.dot(activation_hidden, self.weights_hidden_output) + self.bias_hidden_output
        output = self.activation_function(net_output)
        
        # Backward pass
        output_error = targets - output
        delta_output = output_error * self.activation_function.derivative(output)
        hidden_error = np.dot(delta_output, self.weights_hidden_output.T)
        delta_hidden = hidden_error * self.activation_function.derivative(activation_hidden)
        
        # Update weights and biases
        self.weights_hidden_output += np.dot(activation_hidden.T, delta_output) * self.learning_rate
        self.bias_hidden_output += np.sum(delta_output, axis=0) * self.learning_rate
        self.weights_input_hidden += np.dot(inputs.T, delta_hidden) * self.learning_rate
        self.bias_input_hidden += np.sum(delta_hidden, axis=0) * self.learning_rate

    def train(self, inputs, targets, epochs):
        """Treina a rede neural."""
        for epoch in range(epochs):
            # Forward pass and backpropagation
            self.backpropagation(inputs, targets)
            # Calculate and store the mean squared error for monitoring
            output = self.feedforward(inputs)
            error = np.mean(np.square(targets - output))
            self.train_errors.append(error)
            print(f"Epoch {epoch + 1}/{epochs} - Error: {error:.6f}")

        

    def predict(self, inputs, targets):
        """Faz a previsão da saída para os dados de entrada fornecidos."""
        predictions = self.feedforward(inputs)
        errors = targets - predictions
        return predictions, errors
    def evaluate(self, X_train, y_train, X_test, y_test):
        """Evaluate the neural network using the provided training and testing data."""
        predictions_train, _ = self.predict(X_train, y_train)
        predictions_test, test_errors = self.predict(X_test, y_test)


        # Assuming alarm_triggered is based on the predictions
        threshold = 0.5  # Example threshold
        alarm_triggered_train = (predictions_train > threshold).astype(int)
        alarm_triggered_test = (predictions_test > threshold).astype(int)

        # Calculate evaluation metrics
        accuracy_train = accuracy_score(y_train, predictions_train.round())
        precision_train = precision_score(y_train, predictions_train.round())
        recall_train = recall_score(y_train, predictions_train.round())
        f1_train = f1_score(y_train, predictions_train.round())

        accuracy_test = accuracy_score(y_test, predictions_test.round())
        precision_test = precision_score(y_test, predictions_test.round())
        recall_test = recall_score(y_test, predictions_test.round())
        f1_test = f1_score(y_test, predictions_test.round())

        return predictions_train, predictions_test, test_errors, accuracy_train, precision_train, recall_train, f1_train, accuracy_test, precision_test, recall_test, f1_test, alarm_triggered_test, alarm_triggered_train

    def generate_report(self, predictions, targets, errors, file_name="training_report.pdf"):
        """
        Gera um relatório de treinamento em PDF com os resultados da rede neural.

        Args:
            predictions (numpy.ndarray): Previsões feitas pela rede neural.
            targets (numpy.ndarray): Rótulos de destino reais.
            errors (list): Lista de erros de treinamento.
            file_name (str, optional): Nome do arquivo de saída. Defaults to "training_report.pdf".
        """

        # Gerar o conteúdo do relatório
        doc = SimpleDocTemplate(file_name, pagesize=letter)
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        normal_style = styles['Normal']

        # Título
        title = Paragraph("Relatório de Treinamento", title_style)
        elements = [title]

        # Gráfico de evolução dos erros
        plt.figure(figsize=(4, 2))
        plt.plot(range(1, len(errors) + 1), errors, label='Erro de Treinamento')
        plt.xlabel('Épocas')
        plt.ylabel('Erro')
        plt.title('Evolução do Erro de Treinamento')
        plt.legend()
        plt.grid(True)
        plt.savefig("training_error_plot.png")
        plt.close()

        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Evolução do Erro de Treinamento", normal_style))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("<img src='training_error_plot.png'/>", normal_style))

        # Salvar o PDF
        doc.build(elements)

        # Remover a imagem temporária
        os.remove("training_error_plot.png")

        print("Relatório de treinamento gerado com sucesso.")