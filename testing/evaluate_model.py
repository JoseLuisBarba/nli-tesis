from sklearn.metrics import precision_recall_fscore_support

def calculate_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"Precision": precision, "Recall": recall, "F1 Score": f1}



