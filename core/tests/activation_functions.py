# tests/test_activation_functions.py

import unittest
import numpy as np
from core.activation_functions import Sigmoid, Tanh


class TestSigmoidActivationFunction(unittest.TestCase):
    """Testes unitários para a função de ativação sigmoide."""

    def test_call(self):
        """Testa o cálculo da função sigmoide."""
        sigmoid = Sigmoid()
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertAlmostEqual(sigmoid(1), 0.7310585786300049)
        self.assertAlmostEqual(sigmoid(-1), 0.2689414213699951)

    def test_derivative(self):
        """Testa o cálculo da derivada da função sigmoide."""
        sigmoid = Sigmoid()
        self.assertAlmostEqual(sigmoid.derivative(0), 0.25)
        self.assertAlmostEqual(sigmoid.derivative(1), 0.19661193324148185)
        self.assertAlmostEqual(sigmoid.derivative(-1), 0.19661193324148185)


class TestTanhActivationFunction(unittest.TestCase):
    """Testes unitários para a função de ativação tangente hiperbólica."""

    def test_call(self):
        """Testa o cálculo da função tangente hiperbólica."""
        tanh = Tanh()
        self.assertAlmostEqual(tanh(0), 0.0)
        self.assertAlmostEqual(tanh(1), 0.7615941559557649)
        self.assertAlmostEqual(tanh(-1), -0.7615941559557649)

    def test_derivative(self):
        """Testa o cálculo da derivada da função tangente hiperbólica."""
        tanh = Tanh()
        self.assertAlmostEqual(tanh.derivative(0), 1.0)
        self.assertAlmostEqual(tanh.derivative(1), 0.41997434161402614)
        self.assertAlmostEqual(tanh.derivative(-1), 0.41997434161402614)


if __name__ == "__main__":
    unittest.main()
