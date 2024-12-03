from model.contract_nli_dataset import generate_dataset
from model.load_model import load_model_and_tokenizer
from model.metrics import compute_metrics

from transformers import Trainer, TrainingArguments
import numpy as np
import torch
import torch.nn as nn

class TrainingPipeline:
    def __init__(self,
            model_name: str,
            train_dataframe: str,
            valid_dataframe: str,
            test_dataframe: str,   
        ):
        self.model_name = model_name
        self.model, self.tokenizer = load_model_and_tokenizer(model_name=self.model_name)
        self.train_dataset = generate_dataset(dataframe=train_dataframe, tokenizer=self.tokenizer)
        self.valid_dataset = generate_dataset(dataframe=valid_dataframe, tokenizer=self.tokenizer)
        self.test_dataset = generate_dataset(dataframe=test_dataframe, tokenizer=self.tokenizer)
    
    def compute_class_weights(self, dataset):
        labels = [label for _, label in dataset]
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
        return torch.tensor(class_weights, dtype=torch.float32)
    
    def custom_weighted_cross_entropy_loss(self, weights):
        def loss_fn(logits, labels):
            loss = nn.CrossEntropyLoss(weight=weights)
            return loss(logits, labels)
        return loss_fn
    
    def train(self, ):
        training_args = TrainingArguments(
            output_dir=f"./output/{self.model_name}",
            num_train_epochs=10,  
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch", 
            save_strategy="epoch",  
            save_total_limit=1,  
            load_best_model_at_end=True, 
            metric_for_best_model="eval_f1", 
            greater_is_better=True, 
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        return {
            "trainer":trainer, 
            "model":self.model, 
            "tokenizer":self.tokenizer, 
            "train_dataset":self.train_dataset, 
            "valid_dataset":self.valid_dataset, 
            "test_dataset":self.test_dataset
        }

