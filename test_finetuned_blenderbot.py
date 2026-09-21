from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration
)


MODEL_PATH = "./blenderbot_mental_health_model"


print("Loading fine-tuned tokenizer...")

tokenizer = BlenderbotTokenizer.from_pretrained(
    MODEL_PATH
)


print("Loading fine-tuned BlenderBot...")

model = BlenderbotForConditionalGeneration.from_pretrained(
    MODEL_PATH
)


print("Fine-tuned BlenderBot loaded successfully!")


# --------------------------------------------------
# Test messages
# --------------------------------------------------

test_messages = [
    "I am feeling stressed.",
    "I feel lonely.",
    "I am worried about my exams.",
    "I cannot sleep.",
    "I don't feel confident about myself."
]


# --------------------------------------------------
# Generate responses
# --------------------------------------------------

for user_message in test_messages:

    print()
    print("User:", user_message)

    inputs = tokenizer(
        user_message,
        return_tensors="pt"
    )

    reply_ids = model.generate(
        **inputs,
        max_new_tokens=50
    )

    response = tokenizer.decode(
        reply_ids[0],
        skip_special_tokens=True
    ).strip()

    print("Bot:", response)