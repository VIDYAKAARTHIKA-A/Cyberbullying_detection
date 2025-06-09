from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv(r"C:\Users\Admin\Downloads\cyberbullying_tweets.csv (1)\cyberbullying_tweets.csv")
print("Unique labels in dataset:", df["cyberbullying_type"].unique())

# Define texts and labels
texts = df["tweet_text"].astype(str).tolist()
raw_labels = df["cyberbullying_type"].tolist()

# Updated label map for 6-class classification
label_map = {
    "not_cyberbullying": 0,
    "gender": 1,
    "religion": 2,
    "ethnicity": 3,
    "other_cyberbullying": 4,
    "age": 5
}

# Convert string labels to numeric labels
labels = [label_map[label] for label in raw_labels]

# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# Tokenize the text
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_enc = tokenizer(train_texts, truncation=True, padding=True)
val_enc = tokenizer(val_texts, truncation=True, padding=True)

# Create Hugging Face Datasets
train_dataset = Dataset.from_dict({**train_enc, "labels": train_labels})
val_dataset = Dataset.from_dict({**val_enc, "labels": val_labels})

# Load model for 6-class classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Compute metrics function
def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train and save
trainer.train()
trainer.save_model("./model")
tokenizer.save_pretrained("./model")
