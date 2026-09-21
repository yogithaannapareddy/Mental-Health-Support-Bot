from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "microsoft/DialoGPT-small"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model loaded successfully!")

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Chat ended.")
        break

    # Convert user message into tokens
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt"
    )

    # Generate response
    with torch.no_grad():
        chat_history_ids = model.generate(
            new_input_ids,
            max_length=100,
            pad_token_id=tokenizer.eos_token_id
        )

    # Convert generated tokens back to text
    response = tokenizer.decode(
        chat_history_ids[:, new_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    print("Bot:", response)