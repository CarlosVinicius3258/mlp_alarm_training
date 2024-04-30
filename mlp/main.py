from application.interfaces.controller import ApplicationController
from application.interfaces.report_controller import ReportController

def main():
    controller = ApplicationController()
    report_controller = ReportController()
    
    num_samples = 1000
    num_variables = 5
    
    # Train model
    trained_model = controller.train_model(num_samples, num_variables)
    print("Model trained successfully.")
    
    # Generate evaluation report
    report_controller.generate_evaluation_report(trained_model, num_samples, num_variables)

if __name__ == "__main__":
    main()