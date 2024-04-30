import numpy as np
from infra.persistence.model_trainer import ModelTrainer
from infra.data.data_generator import generate_data
from infra.presentation.report_generator import generate_evaluation_report

class TrainModelUseCase:
    def __init__(self, model_trainer: ModelTrainer):
        self.model_trainer = model_trainer

    def execute(self, num_samples, num_variables):
        # Generate data
        combinations = generate_data(num_samples, num_variables)
        
        # Train the model
        model, history = self.model_trainer.train(combinations)
        predictions = model.predict(combinations)
        alarm_triggered = np.zeros((num_samples, 1))
        # Generate training report
        generate_evaluation_report(combinations, predictions, alarm_triggered, history)
        
        return model