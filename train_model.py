from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
import json
import torch


# --------------------------------------------------
# 1. Load training data
# --------------------------------------------------

with open(
    "data/training_data.json",
    "r",
    encoding="utf-8"
) as file:
    conversations = json.load(file)

print("Conversation pairs loaded:", len(conversations))

dataset = Dataset.from_list(conversations)


# --------------------------------------------------
# 2. Load DialoGPT
# --------------------------------------------------

model_name = "microsoft/DialoGPT-small"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

print("Base model loaded.")


# --------------------------------------------------
# 3. Prepare training data
# --------------------------------------------------

def tokenize_function(example):

    text = (
        "User: "
        + example["user"]
        + " Bot: "
        + example["bot"]
        + tokenizer.eos_token
    )

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=128,
        padding="max_length"
    )

    # Ignore padding tokens during training
    tokens["labels"] = [
        token_id if attention == 1 else -100
        for token_id, attention
        in zip(
            tokens["input_ids"],
            tokens["attention_mask"]
        )
    ]

    return tokens


print("Tokenizing training data...")

tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names
)

print("Tokenization completed!")


# --------------------------------------------------
# 4. Training configuration
# --------------------------------------------------

training_args = TrainingArguments(

    output_dir="./mental_health_model",

    num_train_epochs=1,

    per_device_train_batch_size=1,

    learning_rate=5e-7,

    weight_decay=0.01,

    logging_steps=1,

    save_strategy="no",

    report_to="none",

    fp16=False,

    max_grad_norm=0.5
)


# --------------------------------------------------
# 5. Trainer
# --------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)


# --------------------------------------------------
# 6. Train
# --------------------------------------------------

print()
print("Starting safe fine-tuning...")
print("Please wait...")
print()

trainer.train()


# --------------------------------------------------
# 7. Check model for NaN / Inf
# --------------------------------------------------

print()
print("Checking trained model weights...")

has_nan = False
has_inf = False

for name, parameter in model.named_parameters():

    if torch.isnan(parameter).any():

        print("NaN detected in:", name)
        has_nan = True
        break

    if torch.isinf(parameter).any():

        print("Inf detected in:", name)
        has_inf = True
        break


# --------------------------------------------------
# 8. Save only if model is valid
# --------------------------------------------------

if has_nan or has_inf:

    print()
    print("ERROR: Model contains invalid weights.")
    print("Model was NOT saved.")
    print("Please do not continue to Flask.")

else:

    print("Model weights are valid.")

    print()
    print("Saving fine-tuned model...")

    trainer.save_model("./mental_health_model")
    tokenizer.save_pretrained("./mental_health_model")

    print()
    print("Fine-tuning completed successfully!")
    print("Model saved in: ./mental_health_model")