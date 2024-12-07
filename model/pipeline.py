from model.contract_nli_dataset import generate_dataset
from model.load_model import load_model_and_tokenizer
from model.metrics import compute_metrics
from transformers import PreTrainedModel
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoConfig

import torch
import torch.nn as nn


import torch.nn.functional as F

import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput


class WeightedLoss(nn.Module):
    def __init__(self, weights):
        super(WeightedLoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, labels, num_items_in_batch=None):
        if isinstance(outputs, SequenceClassifierOutput):
            logits = outputs.logits
        else:
            logits = outputs

        self.weights = self.weights.to(logits.device)
        return F.cross_entropy(logits, labels, weight=self.weights, label_smoothing=0.1)



class WeightedFocalLoss(nn.Module):
    def __init__(self, weights, gamma=2.0, label_smoothing=0.1):
        super(WeightedLoss, self).__init__()
        self.weights = weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, outputs, labels, num_items_in_batch=None):
        if isinstance(outputs, SequenceClassifierOutput):
            logits = outputs.logits
        else:
            logits = outputs

        self.weights = self.weights.to(logits.device)
        ce_loss = F.cross_entropy(logits, labels, weight=self.weights, label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class_weights = torch.tensor([0.67903683, 0.85,  2.85017836], dtype=torch.float)
loss_func = WeightedLoss(weights=class_weights)
focal_loss_func = WeightedFocalLoss(weights=class_weights)



#Pipeline
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



    def train(self, ):
        training_args = TrainingArguments(
            output_dir=f"./output/{self.model_name}",
            num_train_epochs=10,  
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
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
            learning_rate=3e-5,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            compute_loss_func=focal_loss_func
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




