# interfaces/neural_network_interface.py

from abc import ABC, abstractmethod
import numpy as np

class NeuralNetworkInterface(ABC):
    """Interface for neural network."""

    @abstractmethod
    def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int):
        """Train the neural network."""
        pass

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Predict outputs for given inputs."""
        pass
