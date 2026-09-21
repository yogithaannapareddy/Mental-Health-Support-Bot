import json
import torch

from datasets import Dataset

from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    Trainer,
    TrainingArguments
)


# --------------------------------------------------
# 1. Load training data
# --------------------------------------------------

with open(
    "data/training_data.json",
    "r",
    encoding="utf-8"
) as file:

    conversations = json.load(file)


print(
    "Conversation pairs loaded:",
    len(conversations)
)


dataset = Dataset.from_list(
    conversations
)


# --------------------------------------------------
# 2. Load BlenderBot
# --------------------------------------------------

MODEL_NAME = "facebook/blenderbot-400M-distill"


print("Loading tokenizer...")

tokenizer = BlenderbotTokenizer.from_pretrained(
    MODEL_NAME
)


print("Loading BlenderBot model...")

model = BlenderbotForConditionalGeneration.from_pretrained(
    MODEL_NAME
)


print("BlenderBot loaded successfully!")


# --------------------------------------------------
# 3. Prepare input and target
# --------------------------------------------------

def tokenize_function(example):

    # User message
    inputs = tokenizer(
        example["user"],
        max_length=64,
        truncation=True,
        padding="max_length"
    )


    # Expected bot response
    targets = tokenizer(
        text_target=example["bot"],
        max_length=96,
        truncation=True,
        padding="max_length"
    )


    # Replace padding tokens with -100
    # so they are ignored during loss calculation

    labels = [
        token if token != tokenizer.pad_token_id else -100
        for token in targets["input_ids"]
    ]


    inputs["labels"] = labels


    return inputs


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

    output_dir="./blenderbot_mental_health_model",

    num_train_epochs=1,

    per_device_train_batch_size=1,

    learning_rate=1e-5,

    weight_decay=0.01,

    logging_steps=1,

    save_strategy="epoch",

    report_to="none",

    fp16=False,

    max_grad_norm=1.0
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
# 6. Fine-tuning
# --------------------------------------------------

print()
print("Starting BlenderBot fine-tuning...")
print("Please wait...")
print()


trainer.train()


# --------------------------------------------------
# 7. Check for invalid weights
# --------------------------------------------------

print()
print("Checking trained model weights...")


has_nan = False
has_inf = False


for name, parameter in model.named_parameters():

    if torch.isnan(parameter).any():

        print(
            "NaN detected in:",
            name
        )

        has_nan = True

        break


    if torch.isinf(parameter).any():

        print(
            "Inf detected in:",
            name
        )

        has_inf = True

        break


# --------------------------------------------------
# 8. Save model only if valid
# --------------------------------------------------

if has_nan or has_inf:

    print()
    print(
        "ERROR: Model contains invalid weights."
    )

    print(
        "Model was NOT saved."
    )

else:

    print(
        "Model weights are valid."
    )


    print()
    print(
        "Saving fine-tuned BlenderBot..."
    )


    trainer.save_model(
        "./blenderbot_mental_health_model"
    )


    tokenizer.save_pretrained(
        "./blenderbot_mental_health_model"
    )


    print()
    print(
        "BlenderBot fine-tuning completed successfully!"
    )


    print(
        "Model saved in: "
        "./blenderbot_mental_health_model"
    )