import torch
import tqdm

label_mapping = {
    0: "NotMentioned",
    1: "Entailment",
    2: "Contradiction"
}

def predict(model, dataset):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    model.to(device)
    preds, labels = [], []

    for example in tqdm.tqdm(dataset, desc="Predicting..."):
        input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        pred_index = torch.argmax(logits, dim=-1).cpu().item()
        preds.append(label_mapping[pred_index])  # Mapear índice a etiqueta
        labels.append(example['label'])

    return preds, labels