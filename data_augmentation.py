import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd

print("Cargando datasets...")
train_dataframe = pd.read_csv("./datos/train.csv")
valid_dataframe = pd.read_csv("./datos/dev.csv")
print(f"Dataset de validación: \n{valid_dataframe.head()}")

model_name = 'eugenesiow/bart-paraphrase'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def paraphrase_sentence(sentence: str):
    inputs = tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True).to(device)
    generated_ids = model.generate(inputs['input_ids'], num_beams=5, num_return_sequences=1, early_stopping=True)
    paraphrased_sentence = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return paraphrased_sentence


def generate_data_augmentation(df, column_name='hypothesis'):
    for idx, row in df.iterrows():
        original_sentence = row[column_name]
        paraphrased_sentence = paraphrase_sentence(original_sentence)
        df.at[idx, column_name] = paraphrased_sentence  # Reemplazar la oración original con la parafraseada
    return df


print("Generando datos aumentados para el conjunto de entrenamiento...")
train_dataframe = generate_data_augmentation(train_dataframe, column_name='hypothesis')

print("Generando datos aumentados para el conjunto de validación...")
valid_dataframe = generate_data_augmentation(valid_dataframe, column_name='hypothesis')


train_dataframe.to_csv('./datos/train_augmented.csv', index=False)
valid_dataframe.to_csv('./datos/dev_augmented.csv', index=False)

print("Datos aumentados guardados en 'train_augmented.csv' y 'dev_augmented.csv'.")

