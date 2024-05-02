import numpy as np


class ActivationFunction:
    """Classe base para funções de ativação."""

    def __call__(self, x):
        """Calcula a função de ativação."""
        raise NotImplementedError

    def derivative(self, x):
        """Calcula a derivada da função de ativação."""
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    """Função de ativação sigmoide."""

    def __call__(self, x):
        """Calcula a função sigmoide."""
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        """Calcula a derivada da função sigmoide."""
        sig_x = self(x)
        return sig_x * (1 - sig_x)


class Tanh(ActivationFunction):
    """Função de ativação tangente hiperbólica."""

    def __call__(self, x):
        """Calcula a função tangente hiperbólica."""
        return np.tanh(x)

    def derivative(self, x):
        """Calcula a derivada da função tangente hiperbólica."""
        tanh_x = self(x)
        return 1 - tanh_x ** 2
