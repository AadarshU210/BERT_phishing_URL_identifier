import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import evaluate

model_path = "Aadi210/BERT_phishing_classifier"

# load dataset (test = validation data)
dataset = load_dataset("shawhin/phishing-site-classification")["test"]

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# load metrics
accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

all_probs = []
all_preds = []
all_labels = []

print("Running evaluation")

for sample in dataset:
    text = sample["text"]
    label = sample["labels"]

    inputs = tokenizer(text, return_tensors = "pt", truncation=True, padding = True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).numpy()[0]

    all_probs.append(probs[1]) #probability of positive side(Not safe)
    all_preds.append(np.argmax(probs))
    all_labels.append(label)

#compute metrics
acc = accuracy.compute(predictions = all_preds, references= all_labels)["accuracy"]
auc = auc_score.compute(prediction_scores = all_probs, references = all_labels)["roc_auc"]

print("\nFinal Evaluation")
print("\nAccuracy:",round(acc, 3))
print("\nROC-AUC:", round(auc, 3))

