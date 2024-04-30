from domain.services import RuleBasedSystem
from infra.data.data_generator import generate_data
from infra.presentation.report_generator import generate_evaluation_report

class ReportController:
    def generate_evaluation_report(self, trained_model, num_samples, num_variables):
        # Generate data
        combinations = generate_data(num_samples, num_variables)
        
        # Determine predictions
        predictions = []
        alarm_triggered = []
        for combination in combinations:
            prediction = trained_model.predict(combination.reshape(1, -1))[0][0]
            predictions.append(int(round(prediction)))
            alarm_triggered.append(RuleBasedSystem.determine_output(combination))
        
        # Generate evaluation report
        generate_evaluation_report(combinations, predictions, alarm_triggered)