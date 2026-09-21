from flask import Flask, request, jsonify, send_from_directory
import sys
import os
import json
from datetime import datetime

from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration
)


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FRONTEND_FOLDER = os.path.join(
    PROJECT_ROOT,
    "frontend"
)

MODEL_FOLDER = os.path.join(
    PROJECT_ROOT,
    "blenderbot_mental_health_model"
)

LOG_FOLDER = os.path.join(
    PROJECT_ROOT,
    "logs"
)

LOG_FILE = os.path.join(
    LOG_FOLDER,
    "chat_sessions.json"
)


# Create logs folder if it does not exist
os.makedirs(
    LOG_FOLDER,
    exist_ok=True
)


# Allow importing input_filter.py
sys.path.append(PROJECT_ROOT)


# ==================================================
# IMPORT INPUT FILTER
# ==================================================

from input_filter import (
    contains_offensive_language,
    get_filter_response
)


# ==================================================
# CREATE FLASK APP
# ==================================================

app = Flask(__name__)


# ==================================================
# LOAD FINE-TUNED BLENDERBOT
# ==================================================

print("Loading fine-tuned BlenderBot tokenizer...")

tokenizer = BlenderbotTokenizer.from_pretrained(
    MODEL_FOLDER
)


print("Loading fine-tuned BlenderBot model...")

model = BlenderbotForConditionalGeneration.from_pretrained(
    MODEL_FOLDER
)


print("Fine-tuned BlenderBot loaded successfully!")


# ==================================================
# SAFETY PHRASES
# ==================================================

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


# ==================================================
# SAFETY MESSAGE CHECK
# ==================================================

def is_safety_message(message):

    message = message.lower()

    for phrase in SAFETY_PHRASES:

        if phrase in message:
            return True

    return False


# ==================================================
# SAFETY RESPONSE
# ==================================================

def get_safety_response():

    return (
        "I'm really sorry that you're going through this. "
        "If you feel that you may hurt yourself or are in immediate danger, "
        "please contact your local emergency service or go to the nearest "
        "emergency department. Please also reach out to someone you trust "
        "and consider contacting a qualified mental-health professional."
    )


# ==================================================
# SAVE CHAT SESSION
# ==================================================

def save_chat_session(user_message, response):

    session = {
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "bot_response": response
    }

    # ----------------------------------------------
    # Read existing sessions
    # ----------------------------------------------

    if os.path.exists(LOG_FILE):

        try:

            with open(
                LOG_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                sessions = json.load(file)

        except (
            json.JSONDecodeError,
            FileNotFoundError
        ):

            sessions = []

    else:

        sessions = []


    # ----------------------------------------------
    # Add new session
    # ----------------------------------------------

    sessions.append(session)


    # ----------------------------------------------
    # Save sessions
    # ----------------------------------------------

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


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/", methods=["GET"])
def home():

    return send_from_directory(
        FRONTEND_FOLDER,
        "index.html"
    )


# ==================================================
# CHAT API
# ==================================================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()


    # ----------------------------------------------
    # Validate request
    # ----------------------------------------------

    if not data or "message" not in data:

        return jsonify({
            "error": "Please provide a message."
        }), 400


    user_message = data["message"].strip()


    if not user_message:

        return jsonify({
            "error": "Message cannot be empty."
        }), 400


    # ==================================================
    # SAFETY CHECK
    # ==================================================

    if is_safety_message(user_message):

        response = get_safety_response()

        save_chat_session(
            user_message,
            response
        )

        return jsonify({
            "user_message": user_message,
            "response": response
        })


    # ==================================================
    # OFFENSIVE LANGUAGE FILTER
    # ==================================================

    if contains_offensive_language(user_message):

        response = get_filter_response()

        save_chat_session(
            user_message,
            response
        )

        return jsonify({
            "user_message": user_message,
            "response": response
        })


    # ==================================================
    # GENERATE RESPONSE USING FINE-TUNED BLENDERBOT
    # ==================================================

    inputs = tokenizer(
        user_message,
        return_tensors="pt",
        max_length=64,
        truncation=True
    )


    reply_ids = model.generate(
        **inputs,
        max_new_tokens=50
    )


    response = tokenizer.decode(
        reply_ids[0],
        skip_special_tokens=True
    ).strip()


    # ==================================================
    # FALLBACK RESPONSE
    # ==================================================

    if not response:

        response = (
            "I'm here to listen. "
            "Would you like to tell me more about how you're feeling?"
        )


    # ==================================================
    # SAVE NORMAL CHAT
    # ==================================================

    save_chat_session(
        user_message,
        response
    )


    # ==================================================
    # RETURN RESPONSE
    # ==================================================

    return jsonify({
        "user_message": user_message,
        "response": response
    })


# ==================================================
# START FLASK SERVER
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )