import numpy as np

def generate_data(num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Gera dados de entrada e saída esperados para treinamento da rede neural.

    Args:
        num_samples (int): Número de amostras de dados a serem geradas.

    Returns:
        tuple[np.ndarray, np.ndarray]: Uma tupla contendo os dados de entrada e saída esperados.
    """
    inputs = np.random.randint(2, size=(num_samples, 5))  # Gera dados binários aleatórios para F, G, H, I, J
    
    # Calcula os valores de saída de acordo com as regras fornecidas
    outputs = np.zeros((num_samples, 1))
    for i, (F, G, H, I, J) in enumerate(inputs):
        if (G == 0 and H == 1) or (F == 0 and H == 1):
            outputs[i] = 1
        elif (F == 1 and G == 1 and H == 1):
            outputs[i] = 0
        elif (I == 1 and J == 0):
            outputs[i] = 1
        elif (F == 0 and G == 0 and H == 0):
            outputs[i] = 1
        else:
            # Adicione aleatoriamente "Alarme" para algumas amostras
            outputs[i] = np.random.randint(2)

    return inputs, outputs
