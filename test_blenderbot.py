from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

MODEL_NAME = "facebook/blenderbot-400M-distill"

print("Loading BlenderBot tokenizer...")

tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)

print("Loading BlenderBot model...")

model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)

print("BlenderBot loaded successfully!")

user_message = "I am feeling stressed."

inputs = tokenizer(
    user_message,
    return_tensors="pt"
)

reply_ids = model.generate(
    **inputs,
    max_new_tokens=40
)

response = tokenizer.decode(
    reply_ids[0],
    skip_special_tokens=True
)

print()
print("User:", user_message)
print("Bot:", response)