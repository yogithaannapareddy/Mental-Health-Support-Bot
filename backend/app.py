from flask import Flask, request, jsonify, send_from_directory
import os
import sys
import json
from datetime import datetime

from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FRONTEND_FOLDER = os.path.join(
    PROJECT_ROOT,
    "frontend"
)

LOG_FOLDER = os.path.join(
    PROJECT_ROOT,
    "logs"
)

LOG_FILE = os.path.join(
    LOG_FOLDER,
    "chat_sessions.json"
)

os.makedirs(LOG_FOLDER, exist_ok=True)


# ============================================================
# INPUT FILTER
# ============================================================

sys.path.append(PROJECT_ROOT)

from input_filter import (
    contains_offensive_language,
    get_filter_response
)


# ============================================================
# HUGGING FACE MODEL
# ============================================================

MODEL_NAME = "yogitha0506/mindcare-blenderbot"

print("========================================")
print("Loading MindCare model...")
print("Model:", MODEL_NAME)
print("========================================")

tokenizer = BlenderbotTokenizer.from_pretrained(
    MODEL_NAME
)

model = BlenderbotForConditionalGeneration.from_pretrained(
    MODEL_NAME
)

model.eval()

print("MindCare model loaded successfully!")


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# SAFETY
# ============================================================

SAFETY_PHRASES = [
    "hurt myself",
    "harm myself",
    "kill myself",
    "end my life",
    "suicide",
    "want to die",
    "don't want to live",
    "do not want to live"
]


def contains_safety_phrase(message):

    message = message.lower()

    for phrase in SAFETY_PHRASES:

        if phrase in message:
            return True

    return False


def get_safety_response():

    return (
        "I'm really sorry that you're going through this. "
        "You don't have to face this alone. "
        "If you are in immediate danger, please contact your "
        "local emergency services or go to the nearest emergency "
        "department. You can also reach out to someone you trust "
        "and stay with them. A qualified mental-health professional "
        "can provide appropriate support."
    )


# ============================================================
# SESSION LOGGING
# ============================================================

def save_chat_session(user_message, response):

    session = {
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "bot_response": response
    }

    try:

        if os.path.exists(LOG_FILE):

            with open(
                LOG_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                sessions = json.load(file)

        else:

            sessions = []

    except Exception:

        sessions = []


    sessions.append(session)


    with open(
        LOG_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            sessions,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# FRONTEND
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_FOLDER,
        "index.html"
    )


@app.route("/style.css")
def style():

    return send_from_directory(
        FRONTEND_FOLDER,
        "style.css"
    )


# ============================================================
# CHAT API
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():

    print("Received /chat request")

    try:

        data = request.get_json(silent=True)

        print("Received data:", data)

        if not data:

            return jsonify({
                "response": "Please enter a message."
            }), 400


        user_message = data.get(
            "message",
            ""
        ).strip()


        print("User message:", user_message)


        if not user_message:

            return jsonify({
                "response": "Please enter a message."
            }), 400


        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if contains_safety_phrase(user_message):

            response = get_safety_response()

            save_chat_session(
                user_message,
                response
            )

            return jsonify({
                "response": response
            })


        # ----------------------------------------------------
        # OFFENSIVE LANGUAGE CHECK
        # ----------------------------------------------------

        if contains_offensive_language(user_message):

            response = get_filter_response()

            save_chat_session(
                user_message,
                response
            )

            return jsonify({
                "response": response
            })


        # ----------------------------------------------------
        # TOKENIZATION
        # ----------------------------------------------------

        inputs = tokenizer(
            user_message,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )


        # ----------------------------------------------------
        # MODEL GENERATION
        # ----------------------------------------------------

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=4,
            early_stopping=True
        )


        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        response = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()


        if not response:

            response = (
                "I'm here to listen. "
                "Would you like to tell me a little more "
                "about how you're feeling?"
            )


        print("Bot response:", response)


        # ----------------------------------------------------
        # SAVE SESSION
        # ----------------------------------------------------

        save_chat_session(
            user_message,
            response
        )


        return jsonify({
            "response": response
        })


    except Exception as error:

        print("========================================")
        print("CHAT ERROR:")
        print(error)
        print("========================================")

        return jsonify({
            "response": (
                "I'm sorry, something went wrong. "
                "Please try again."
            )
        }), 500


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )