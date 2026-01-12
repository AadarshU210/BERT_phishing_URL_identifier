from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments  
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding

# loading dataset
dataset_dict = load_dataset("shawhin/phishing-site-classification")

# defining pre-trained model path
model_path = "google-bert/bert-base-uncased"

# load model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# load model with binary classification head

id2label = {0: "Safe", 1: "Not Safe"}
label2id = {"Safe": 0, "Not Safe": 1}

model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, id2label=id2label, label2id=label2id)

# freeze all base model parameters
for name, param in model.base_model.named_parameters():
    param.requires_grad = False

# unfreeze base model pooling layers
for name, param in model.base_model.named_parameters():
    if "pooler" in name:
        param.requires_grad = True

# define text preprocessing function
def preprocess_text(examples):
    # returning tokenized text with truncation
    return tokenizer(examples["text"], truncation=True, padding=True)

# preprocess all dataset
tokenized_dataset = dataset_dict.map(preprocess_text, batched=True)

# creating data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# load metrics
accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    # get_predictions
    predictions, labels = eval_pred

    # apply softmax to get probabilities
    probabilities = np.exp(predictions)/np.exp(predictions).sum(-1, keepdims=True)

    # use probablities of the positive class for ROC AUC
    positive_class_probs = probabilities[:, 1]

    # compute auc
    auc = np.round(auc_score.compute(prediction_scores=positive_class_probs, references=labels)["roc_auc"],3)

    # predict most probable class
    predicted_classes = np.argmax(predictions, axis=1)
    # compute accuracy
    acc = np.round(accuracy.compute(predictions=predicted_classes, references=labels)['accuracy'], 3)

    # return metrics
    return {"accuracy": acc, "roc_auc": auc}


# hyperparameters

lr = 2e-4
batch_size = 8
num_epochs = 10

training_args = TrainingArguments(
    output_dir = "output",
    learning_rate = lr,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    num_train_epochs = num_epochs,
    logging_strategy = "epoch",
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
)

trainer = Trainer(
    model =model,
    args = training_args,
    train_dataset = tokenized_dataset["train"],
    eval_dataset = tokenized_dataset["test"],
    tokenizer = tokenizer,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
)

trainer.train()

# optional
# model.push_to_hub("your-username/your-model-name")
# tokenizer.push_to_hub("your-username/your-model-name")