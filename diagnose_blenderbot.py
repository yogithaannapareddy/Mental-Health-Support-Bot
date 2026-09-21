import torch
from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration
)

MODEL_NAME = "facebook/blenderbot-400M-distill"

print("Loading tokenizer...")
tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)

print("Model loaded successfully.")

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

model.train()

# Very small learning rate
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-9,
    eps=1e-8
)

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

print("Running backward pass...")

loss.backward()

print("Backward pass completed.")

# Check gradients
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

# Clip gradients
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=0.1
)

print("Gradient clipping completed.")

# Save original shared weight for comparison
before = model.shared.weight.detach().clone()

print("Running optimizer step...")

optimizer.step()

print("Optimizer step completed.")

after = model.shared.weight.detach()

print("Checking model weights...")

if torch.isnan(after).any():
    print("❌ NaN detected after optimizer step.")
elif torch.isinf(after).any():
    print("❌ Inf detected after optimizer step.")
else:
    print("✅ Model weights are VALID after optimizer step.")

difference = torch.abs(after - before).max().item()

print("Maximum weight change:", difference)

print()
print("Diagnostic 2 completed.")