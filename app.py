import streamlit as st
import json
import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# LOAD INTENTS
# ==========================================

with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)


# ==========================================
# PREPARE TRAINING DATA
# ==========================================

training_sentences = []
training_intents = []

for intent in data["intents"]:

    for pattern in intent.get("patterns", []):

        training_sentences.append(pattern)
        training_intents.append(intent)


# ==========================================
# CREATE TF-IDF MODEL
# ==========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    training_sentences
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Mental Health Support Bot",
    page_icon="🧠",
    layout="centered"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 35px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# TITLE
# ==========================================

st.markdown(
    '<div class="title">🧠 Mental Health Support Bot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">A safe space to share what is on your mind.</div>',
    unsafe_allow_html=True
)


# ==========================================
# DISCLAIMER
# ==========================================

st.info(
    "💙 This bot provides general emotional support and is not "
    "a replacement for a qualified mental-health professional."
)


# ==========================================
# CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# ==========================================
# USER INPUT
# ==========================================

user_input = st.chat_input(
    "How are you feeling today?"
)


# ==========================================
# PROCESS USER MESSAGE
# ==========================================

if user_input:

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    user_input_lower = user_input.lower()


    # ==========================================
    # SAFETY CHECK
    # ==========================================

    safety_keywords = [
        "hurt myself",
        "harm myself",
        "kill myself",
        "end my life",
        "suicide",
        "want to die",
        "don't want to live",
        "do not want to live"
    ]

    safety_detected = False

    for keyword in safety_keywords:

        if re.search(
            r"\b" + re.escape(keyword) + r"\b",
            user_input_lower
        ):

            safety_detected = True
            break


    # ==========================================
    # SAFETY RESPONSE
    # ==========================================

    if safety_detected:

        response = (
            "I'm really sorry that you're going through this. "
            "Your safety is important. If you feel you may hurt "
            "yourself or are in immediate danger, please contact "
            "your local emergency service or go to the nearest "
            "emergency department. If possible, stay with someone "
            "you trust and tell them what you're experiencing. "
            "A qualified mental-health professional can also "
            "provide appropriate support."
        )


    # ==========================================
    # NLP INTENT DETECTION
    # ==========================================

    else:

        # Convert user message into TF-IDF vector
        user_vector = vectorizer.transform(
            [user_input]
        )

        # Compare user message with training sentences
        similarities = cosine_similarity(
            user_vector,
            tfidf_matrix
        )[0]

        # Find the most similar sentence
        best_index = similarities.argmax()

        best_score = similarities[best_index]


        # ==========================================
        # CONFIDENCE THRESHOLD
        # ==========================================

        if best_score >= 0.25:

            best_intent = training_intents[best_index]

            response = random.choice(
                best_intent["responses"]
            )

        else:

            response = (
                "I'm listening. Please tell me a little more "
                "about what you're experiencing."
            )


    # ==========================================
    # DISPLAY BOT RESPONSE
    # ==========================================

    with st.chat_message("assistant"):
        st.write(response)


    # Save bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })