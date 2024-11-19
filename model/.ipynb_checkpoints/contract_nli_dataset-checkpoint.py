import torch
import pandas as pd 
from model.labels import label_mapping
from model.tokenize import tokenize_data

class ContractNLIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)
    

class ContractNLIDatasetTest(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])
    


def generate_dataset(dataframe, tokenizer):
    labels = [ label_mapping[label] for label in dataframe["label"].to_list()]
    encodings = tokenize_data(data=dataframe, tokenizer=tokenizer)
    return ContractNLIDataset(encodings, labels)