# %% [code]
!pip install -q transformers datasets peft accelerate evaluate

# %% [code]
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import numpy as np

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## 1. Load Dataset (IMDB)

# %% [code]
# Load the IMDB sentiment dataset (binary: 0 = negative, 1 = positive)
raw_datasets = load_dataset("imdb")

print(raw_datasets)

# Optional: for faster experiments, you can subsample the dataset.
# Comment these lines out if you want to use the full dataset.
small_train = raw_datasets["train"].shuffle(seed=42).select(range(4000))
small_test = raw_datasets["test"].shuffle(seed=42).select(range(2000))

raw_datasets = {
    "train": small_train,
    "test": small_test
}

# Create a validation split from the training set (e.g. 90% train / 10% val)
split = raw_datasets["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]
test_dataset = raw_datasets["test"]

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))

# %% [markdown]
# ## 2. Tokenizer & Preprocessing

# %% [code]
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_length = 256

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length
    )

encoded_train = train_dataset.map(preprocess_function, batched=True)
encoded_val = val_dataset.map(preprocess_function, batched=True)
encoded_test = test_dataset.map(preprocess_function, batched=True)

# Tell Trainer which columns are inputs/labels
encoded_train = encoded_train.remove_columns(["text"])
encoded_val = encoded_val.remove_columns(["text"])
encoded_test = encoded_test.remove_columns(["text"])

encoded_train = encoded_train.rename_column("label", "labels")
encoded_val = encoded_val.rename_column("label", "labels")
encoded_test = encoded_test.rename_column("label", "labels")

encoded_train.set_format("torch")
encoded_val.set_format("torch")
encoded_test.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %% [markdown]
# ## 3. Load Base Model with Classification Head

# %% [code]
num_labels = 2
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

base_model.to(device)

# %% [markdown]
# ## 4. Apply LoRA with PEFT

# %% [code]
# LoRA configuration (you can mention these in the README)
lora_config = LoraConfig(
    r=8,                     # Rank
    lora_alpha=16,           # LoRA scaling
    lora_dropout=0.1,        # Dropout for LoRA layers
    task_type=TaskType.SEQ_CLS,
)

peft_model = get_peft_model(base_model, lora_config)
peft_model.to(device)

# Show how many parameters are trainable with LoRA
def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Trainable params: {trainable}")
    print(f"Total params: {total}")
    print(f"Trainable %: {100 * trainable / total:.4f}%")

print_trainable_parameters(peft_model)

# %% [markdown]
# ## 5. Metrics (Accuracy, F1, Precision, Recall)

# %% [code]
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    prec = precision_metric.compute(predictions=preds, references=labels, average="weighted")["precision"]
    rec = recall_metric.compute(predictions=preds, references=labels, average="weighted")["recall"]
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
    }

# %% [markdown]
# ## 6. Training Setup (Trainer + LoRA model)

# %% [code]
batch_size = 16
num_epochs = 2  # You can increase to 3–5 if you have time/compute

training_args = TrainingArguments(
    output_dir="./results-lora-imdb",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=2e-4,         # A bit higher LR since only LoRA params are trained
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="none",           # Turn off wandb, etc.
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %% [markdown]
# ## 7. Fine-Tune the LoRA-Enabled Model

# %% [code]
train_result = trainer.train()
print("Training complete.")

# Evaluate on validation set (best model)
val_metrics = trainer.evaluate(encoded_val)
print("Validation metrics:", val_metrics)

# %% [markdown]
# ## 8. Final Evaluation on Test Set

# %% [code]
test_metrics = trainer.evaluate(encoded_test)
print("Test metrics:", test_metrics)

# You can print key metrics cleanly for the README
print("\n=== Final Test Results (IMDB + LoRA) ===")
print(f"Accuracy:  {test_metrics['eval_accuracy']:.4f}")
print(f"F1-score:  {test_metrics['eval_f1']:.4f}")
print(f"Precision: {test_metrics['eval_precision']:.4f}")
print(f"Recall:    {test_metrics['eval_recall']:.4f}")
