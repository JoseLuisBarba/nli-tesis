import torch
import pandas as pd
from testing.load_model import load_model_and_tokenizer
from testing.load_dataset import prepare_dataset
from testing.predict import predict
from testing.evaluate_model import calculate_metrics
import tqdm




class ContractNLITestPipeline:
    def __init__(self,
            model_names = ["MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", ],
            dataset_name = "./datos/test.csv",
            example_size = None
        ):
        self.model_names = model_names
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.data_test = pd.read_csv("./datos/test.csv")
        if example_size:
            self.data_test = self.data_test.iloc[0:example_size,]

    def measure_model(self, name, dataset):
        model, tokenizer = load_model_and_tokenizer(name)
        dataset = prepare_dataset(self.data_test, tokenizer)
        preds, labels = predict(model, dataset)
        metrics = calculate_metrics(preds, labels)
        return metrics 

    def __call__(self, *args, **kwds):
        metrics = []
        names = []
        for model_name in tqdm.tqdm(self.model_names, desc="Predicting..."):
            model_metrics = self.measure_model(model_name, dataset=self.data_test)
            metrics.append(model_metrics)
            names.append(model_name)
        return pd.DataFrame(metrics, index=names)



