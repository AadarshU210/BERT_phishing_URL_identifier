from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "Aadi210/BERT_phishing_classifier"

id2label = {0:"Safe", 1:"Not Safe"}

app = FastAPI(title="Phishing URL detection API")

print("Loading Model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()


class TextInput(BaseModel):
    text:str

@app.post("/predict")
def predict(input: TextInput):
    inputs = tokenizer(input.text, return_tensors = "pt", truncation = True, padding = True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]

    pred = int(np.argmax(probs))
    confidence = float(probs[pred])

    return {
        "text": input.text,
        "prediction":id2label[pred],
        "confidence":round(confidence, 3)
    }