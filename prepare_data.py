import json

input_file = "data/mental_health_dialogues.txt"
output_file = "data/training_data.json"

conversations = []

with open(input_file, "r", encoding="utf-8") as file:
    lines = [line.strip() for line in file if line.strip()]

for i in range(0, len(lines) - 1, 2):

    user_line = lines[i]
    bot_line = lines[i + 1]

    if user_line.startswith("User:") and bot_line.startswith("Bot:"):

        user_message = user_line.replace("User:", "").strip()
        bot_message = bot_line.replace("Bot:", "").strip()

        conversations.append({
            "user": user_message,
            "bot": bot_message
        })

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(conversations, file, indent=4, ensure_ascii=False)

print("Training data prepared successfully!")
print("Total conversation pairs:", len(conversations))
print("Saved to:", output_file)