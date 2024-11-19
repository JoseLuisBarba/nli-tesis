from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    return model, tokenizer







