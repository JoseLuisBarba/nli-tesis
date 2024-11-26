import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd
from tqdm import tqdm  

print("Cargando datasets...")
train_dataframe = pd.read_csv("./datos/train.csv")
valid_dataframe = pd.read_csv("./datos/dev.csv")
print(f"Dataset de validación: \n{valid_dataframe.head()}")

model_name = 'eugenesiow/bart-paraphrase'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


batch_size = 8

def paraphrase_batch(sentences: list):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    generated_ids = model.generate(
        inputs['input_ids'], 
        num_beams=10,  
        num_return_sequences=1,  
        max_length=35,  
        early_stopping=True, 
        no_repeat_ngram_size=2,  
        temperature=1.2, 
        top_k=50,  
        top_p=0.95  
    )
    paraphrased_sentences = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return paraphrased_sentences

def generate_data_augmentation(df, column_name='hypothesis'):
    new_sentences = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch_sentences = df[column_name].iloc[i:i+batch_size].tolist()
        paraphrased_batch = paraphrase_batch(batch_sentences)
        new_sentences.extend(paraphrased_batch)
    
    df[column_name] = new_sentences
    return df

print("Generando datos aumentados para el conjunto de entrenamiento...")
train_dataframe = generate_data_augmentation(train_dataframe, column_name='hypothesis')

print("Generando datos aumentados para el conjunto de validación...")
valid_dataframe = generate_data_augmentation(valid_dataframe, column_name='hypothesis')

train_dataframe.to_csv('./datos/train_augmented.csv', index=False)
valid_dataframe.to_csv('./datos/dev_augmented.csv', index=False)

print("Datos aumentados guardados en 'train_augmented.csv' y 'dev_augmented.csv'.")



