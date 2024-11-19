from datasets import Dataset


def prepare_dataset(df, tokenizer):
    def tokenize_examples(examples):
        # Tokenizar pares de texto e hipótesis
        tokenized = tokenizer(examples['text'], examples['hypothesis'], padding=True, truncation=True, return_tensors="pt")
        # Asegurarse de que todos los resultados sean tensores
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'label': examples['label']  # Esto debe ser un tensor o lista compatible
        }

    # Convertir DataFrame en Dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(tokenize_examples, batched=True)
    return dataset