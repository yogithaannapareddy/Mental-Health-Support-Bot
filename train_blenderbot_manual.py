import json
import torch

from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration
)


# ==================================================
# SETTINGS
# ==================================================

MODEL_NAME = "facebook/blenderbot-400M-distill"

DATA_FILE = "data/training_data.json"

OUTPUT_FOLDER = "blenderbot_mental_health_model"

LEARNING_RATE = 1e-9

MAX_GRAD_NORM = 0.1

EPOCHS = 1


# ==================================================
# LOAD TRAINING DATA
# ==================================================

print("Loading training data...")

with open(DATA_FILE, "r", encoding="utf-8") as file:
    conversations = json.load(file)

print(
    "Training examples:",
    len(conversations)
)


# ==================================================
# LOAD TOKENIZER
# ==================================================

print()
print("Loading tokenizer...")

tokenizer = BlenderbotTokenizer.from_pretrained(
    MODEL_NAME
)


# ==================================================
# LOAD MODEL
# ==================================================

print("Loading BlenderBot model...")

model = BlenderbotForConditionalGeneration.from_pretrained(
    MODEL_NAME
)

print("BlenderBot loaded successfully.")


# ==================================================
# TRAINING
# ==================================================

model.train()


for epoch in range(EPOCHS):

    print()
    print(
        "========== EPOCH",
        epoch + 1,
        "=========="
    )

    for index, conversation in enumerate(conversations):

        user_text = conversation["user"]

        bot_text = conversation["bot"]


        # ------------------------------------------
        # Tokenize user input
        # ------------------------------------------

        inputs = tokenizer(
            user_text,
            return_tensors="pt",
            max_length=64,
            truncation=True
        )


        # ------------------------------------------
        # Tokenize expected response
        # ------------------------------------------

        targets = tokenizer(
            text_target=bot_text,
            return_tensors="pt",
            max_length=96,
            truncation=True
        )


        labels = targets["input_ids"]


        # ------------------------------------------
        # Clear previous gradients
        # ------------------------------------------

        model.zero_grad()


        # ------------------------------------------
        # Forward pass
        # ------------------------------------------

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )


        loss = outputs.loss


        # ------------------------------------------
        # Check loss
        # ------------------------------------------

        if torch.isnan(loss) or torch.isinf(loss):

            print()
            print(
                "ERROR: Invalid loss at example",
                index + 1
            )

            print("Training stopped.")

            exit()


        # ------------------------------------------
        # Backward pass
        # ------------------------------------------

        loss.backward()


        # ------------------------------------------
        # Check gradients
        # ------------------------------------------

        invalid_gradient = False


        for name, parameter in model.named_parameters():

            if parameter.grad is not None:

                if torch.isnan(parameter.grad).any():

                    print(
                        "NaN gradient detected:",
                        name
                    )

                    invalid_gradient = True

                    break


                if torch.isinf(parameter.grad).any():

                    print(
                        "Inf gradient detected:",
                        name
                    )

                    invalid_gradient = True

                    break


        if invalid_gradient:

            print("Training stopped.")

            exit()


        # ------------------------------------------
        # Gradient clipping
        # ------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=MAX_GRAD_NORM
        )


        # ------------------------------------------
        # Manual weight update
        # ------------------------------------------

        with torch.no_grad():

            for name, parameter in model.named_parameters():

                if parameter.grad is not None:

                    parameter -= (
                        LEARNING_RATE *
                        parameter.grad
                    )


        # ------------------------------------------
        # Check model weights
        # ------------------------------------------

        invalid_weight = False


        for name, parameter in model.named_parameters():

            if torch.isnan(parameter).any():

                print()
                print(
                    "NaN weight detected:",
                    name
                )

                invalid_weight = True

                break


            if torch.isinf(parameter).any():

                print()
                print(
                    "Inf weight detected:",
                    name
                )

                invalid_weight = True

                break


        if invalid_weight:

            print()
            print(
                "Training stopped at example:",
                index + 1
            )

            exit()


        # ------------------------------------------
        # Print progress
        # ------------------------------------------

        print(
            "Example",
            index + 1,
            "/",
            len(conversations),
            "| Loss:",
            round(loss.item(), 4)
        )


# ==================================================
# SAVE MODEL
# ==================================================

print()
print("Training completed successfully.")


print("Saving model...")


model.save_pretrained(
    OUTPUT_FOLDER
)


tokenizer.save_pretrained(
    OUTPUT_FOLDER
)


print()
print("Fine-tuned BlenderBot saved successfully!")

print(
    "Model folder:",
    OUTPUT_FOLDER
)