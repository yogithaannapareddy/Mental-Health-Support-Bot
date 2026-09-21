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
# Prepare one small training example
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
# Training mode
# --------------------------------------------------

model.train()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-9,
    eps=1e-8
)


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

print()
print("Running forward pass...")


outputs = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    labels=labels
)


loss = outputs.loss


print("Loss:", loss.item())


if torch.isnan(loss) or torch.isinf(loss):

    print("ERROR: Invalid loss.")

    exit()


# --------------------------------------------------
# Backward pass
# --------------------------------------------------

print()
print("Running backward pass...")


loss.backward()


print("Backward pass completed.")


# --------------------------------------------------
# Check gradients
# --------------------------------------------------

print()
print("Checking gradients...")


bad_gradient = False


for name, parameter in model.named_parameters():

    if parameter.grad is not None:

        if torch.isnan(parameter.grad).any():

            print("NaN gradient:", name)

            bad_gradient = True

            break


        if torch.isinf(parameter.grad).any():

            print("Inf gradient:", name)

            bad_gradient = True

            break


if bad_gradient:

    print("ERROR: Invalid gradient detected.")

    exit()


print("Gradients are valid.")


# --------------------------------------------------
# Gradient clipping
# --------------------------------------------------

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=0.1
)


print("Gradient clipping completed.")


# --------------------------------------------------
# Check weights BEFORE optimizer step
# --------------------------------------------------

print()
print("Checking weights before optimizer step...")


bad_before = False


for name, parameter in model.named_parameters():

    if torch.isnan(parameter).any():

        print("NaN before optimizer:", name)

        bad_before = True

        break

    if torch.isinf(parameter).any():

        print("Inf before optimizer:", name)

        bad_before = True

        break


if bad_before:

    print("ERROR: Model already has invalid weights.")

    exit()


print("Weights are valid before optimizer step.")


# --------------------------------------------------
# Optimizer step
# --------------------------------------------------

print()
print("Running optimizer step...")


optimizer.step()


print("Optimizer step completed.")


# --------------------------------------------------
# Check weights AFTER optimizer step
# --------------------------------------------------

print()
print("Checking weights after optimizer step...")


bad_after = False


for name, parameter in model.named_parameters():

    if torch.isnan(parameter).any():

        print("❌ NaN detected after optimizer step:", name)

        bad_after = True

        break

    if torch.isinf(parameter).any():

        print("❌ Inf detected after optimizer step:", name)

        bad_after = True

        break


if not bad_after:

    print("✅ Model weights are VALID after optimizer step.")


print()
print("Diagnostic 2 completed.")