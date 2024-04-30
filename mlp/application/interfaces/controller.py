from domain.services import RuleBasedSystem
from infra.persistence.model_trainer import ModelTrainer
from application import TrainModelUseCase

class ApplicationController:
    def train_model(self, num_samples, num_variables):
        model_trainer = ModelTrainer()
        train_model_use_case = TrainModelUseCase(model_trainer)
        trained_model = train_model_use_case.execute(num_samples, num_variables)
        return trained_model