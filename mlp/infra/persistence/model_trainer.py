import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class ModelTrainer:
    def train(self, combinations):
        model = Sequential()
        model.add(Dense(10, input_dim=combinations.shape[1], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(combinations, combinations[:, 0], epochs=100, batch_size=1, verbose=0)
        return model, history