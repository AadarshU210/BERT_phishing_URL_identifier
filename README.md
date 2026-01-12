## Trained Model

The fine-tuned model is available on HuggingFace:

https://huggingface.co/Aadi210/BERT_phishing_classifier

## API Inference

This project provides a FastAPI-based inference service for real-time phishing detection.

### Run the API

uvicorn src.predict_api:app --reload

The API will be available at:
http://127.0.0.1:8000

## How to test

Open:
http://127.0.0.1:8000/docs

Use the /predict endpoint with JSON input:

{
  "text": "<URL-to-be-tested>"
}

Example response:

{
  "prediction": "Not Safe",
  "confidence": 0.97
}

## Limitations

This model is trained on a phishing text dataset and may flag legitimate brand-related messages as risky.
It should be used as a risk scoring component, not as a final security decision system.
