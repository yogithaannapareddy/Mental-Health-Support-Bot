import torch

from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration
)


MODEL_NAME = "facebook/blenderbot-400M-distill"


print("Loading tokenizer...")

tokenizer = BlenderbotTokenizer.from_pretrained(
    MODEL_NAME
)


print("Loading model...")

model = BlenderbotForConditionalGeneration.from_pretrained(
    MODEL_NAME
)


print("Model loaded successfully.")


# --------------------------------------------------
# Training example
# --------------------------------------------------

user_text = "I am feeling stressed."

bot_text = (
    "I'm sorry you're feeling stressed. "
    "Would you like to talk about what is causing the stress?"
)


inputs = tokenizer(
    user_text,
    return_tensors="pt",
    max_length=64,
    truncation=True
)


targets = tokenizer(
    text_target=bot_text,
    return_tensors="pt",
    max_length=96,
    truncation=True
)


labels = targets["input_ids"]


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

model.train()

print()
print("Running forward pass...")


outputs = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    labels=labels
)


loss = outputs.loss


print("Loss:", loss.item())


# --------------------------------------------------
# Backward pass
# --------------------------------------------------

print()
print("Running backward pass...")


loss.backward()


print("Backward pass completed.")


# --------------------------------------------------
# Check gradient
# --------------------------------------------------

print()
print("Checking gradients...")


for name, parameter in model.named_parameters():

    if parameter.grad is not None:

        if torch.isnan(parameter.grad).any():

            print("NaN gradient:", name)
            exit()

        if torch.isinf(parameter.grad).any():

            print("Inf gradient:", name)
            exit()


print("All gradients are valid.")


# --------------------------------------------------
# Gradient clipping
# --------------------------------------------------

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=0.1
)


print("Gradient clipping completed.")


# --------------------------------------------------
# Manual update
# --------------------------------------------------

print()
print("Performing manual weight update...")


learning_rate = 1e-9


with torch.no_grad():

    for name, parameter in model.named_parameters():

        if parameter.grad is not None:

            parameter -= learning_rate * parameter.grad


print("Manual update completed.")


# --------------------------------------------------
# Check weights
# --------------------------------------------------

print()
print("Checking model weights...")


bad_weights = False


for name, parameter in model.named_parameters():

    if torch.isnan(parameter).any():

        print("❌ NaN detected:", name)

        bad_weights = True

        break


    if torch.isinf(parameter).any():

        print("❌ Inf detected:", name)

        bad_weights = True

        break


if not bad_weights:

    print("✅ All model weights are VALID.")


print()
print("Manual update test completed.")