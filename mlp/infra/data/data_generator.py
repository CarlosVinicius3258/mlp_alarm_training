import numpy as np

def generate_data(num_samples, num_variables):
    return np.random.randint(0, 2, size=(num_samples, num_variables))