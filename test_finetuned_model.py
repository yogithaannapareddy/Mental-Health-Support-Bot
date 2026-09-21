from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "./mental_health_model"

print("Loading fine-tuned model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")

test_message = "I am feeling stressed"

prompt = "User: " + test_message + " Bot:"

input_ids = tokenizer.encode(
    prompt,
    return_tensors="pt"
)
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=20,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )

response = tokenizer.decode(
    output_ids[0][input_ids.shape[-1]:],
    skip_special_tokens=True
).strip()

print()
print("User:", test_message)
print("Bot:", response)