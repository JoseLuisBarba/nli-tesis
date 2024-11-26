from model.pipeline import TrainingPipeline
from model.evaluation import EvaluationPipeline
import pandas as pd
import json

print("Cargando datasets...")

train_dataframe = pd.read_csv("./datos/train.csv")
valid_dataframe = pd.read_csv("./datos/dev.csv")
test_dataframe = pd.read_csv("./datos/dev.csv")

print("Datasets Cargados.")

print("Entrenando modelo...")

name="tasksource/deberta-base-long-nli"

pipeline = TrainingPipeline(
    model_name=name,
    train_dataframe=train_dataframe,
    valid_dataframe=valid_dataframe,
    test_dataframe=test_dataframe
)

trained = pipeline.train()

print("Evaluando modelo modelo...")

evaluation_pipeline = EvaluationPipeline(
    model=trained["model"], 
    tokenizer=trained["tokenizer"], 
    test_dataset=trained["test_dataset"]
)

evaluation_results = evaluation_pipeline.evaluate()
print("Métricas:", evaluation_results["metrics"])
print("Reporte de clasificación:\n", evaluation_results["classification_report"]) 


with open(f"evaluation_results_deberta-base-long-nli.json", "w") as json_file:
    json.dump(evaluation_results["metrics"], json_file, indent=4)
print("Resultados guardados en 'evaluation_results.txt' y 'evaluation_results.json'.")