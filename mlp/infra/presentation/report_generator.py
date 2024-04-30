from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import io

def generate_evaluation_report(combinations, predictions, alarm_triggered, history):
    # Criar um documento PDF
    doc = SimpleDocTemplate("evaluation_report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Adicionar o título ao relatório
    title = Paragraph("Evaluation Report", styles['Title'])
    doc.build([title])
    
    # Adicionar texto descritivo
    content = []
    content.append(Paragraph("Below are the results of the evaluation:", styles['Normal']))
    content.append(Spacer(1, 12))
    
    # Adicionar os detalhes das previsões e alarmes
    data = []
    data.append(["Sample", "Input Values", "Prediction", "Alarm Triggered"])
    data.append(["-"*6, "-"*24, "-"*10, "-"*15])
    for i, (combination, prediction, alarm) in enumerate(zip(combinations, predictions, alarm_triggered), 1):
        input_values = ", ".join(map(str, combination))
        prediction_text = "Alarm Triggered" if prediction == 1 else "No Alarm"
        alarm_text = "Yes" if alarm == 1 else "No"
        data.append([str(i), input_values, prediction_text, alarm_text])
    
    # Adicionar a tabela ao relatório
    table_style = [('GRID', (0, 0), (-1, -1), 1, (0, 0, 0))]
    content.append(Spacer(1, 12))
    content.append(Paragraph("Evaluation Details:", styles['Heading2']))
    content.append(Spacer(1, 12))
    content.append(create_table(data, table_style))
    
    # Adicionar o gráfico de métricas de treinamento
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Training Metrics')
    plt.legend()
    
    # Salvar o gráfico em um buffer de memória
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image(buf)
    content.append(Spacer(1, 24))
    content.append(Paragraph("Training Metrics:", styles['Heading2']))
    content.append(Spacer(1, 12))
    content.append(Paragraph("The following plot shows the training metrics over epochs:", styles['Normal']))
    content.append(Spacer(1, 6))
    content.append(img)
    
    # Adicionar o conteúdo ao documento
    doc.build(content)
    
    print("Evaluation report generated successfully.")

def create_table(data, style):
    from reportlab.platypus import Table, TableStyle
    return Table(data, style=TableStyle(style))
