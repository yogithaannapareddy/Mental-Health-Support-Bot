import re


# Basic list of offensive words
OFFENSIVE_WORDS = {
    "idiot",
    "stupid",
    "fool",
    "dumb",
    "shut up"
}


def contains_offensive_language(text):
    """
    Checks whether the user's message contains
    any basic offensive words.
    """

    text = text.lower()

    for word in OFFENSIVE_WORDS:

        # Match complete words instead of partial words
        pattern = r"\b" + re.escape(word) + r"\b"

        if re.search(pattern, text):
            return True

    return False


def get_filter_response():
    """
    Response shown when offensive language is detected.
    """

    return (
        "I'd like to keep our conversation respectful. "
        "If you'd like to talk about how you're feeling, "
        "I'm here to listen."
    )


# Test the filter
if __name__ == "__main__":

    test_messages = [
        "I am feeling stressed",
        "You are stupid",
        "I feel lonely",
        "This is dumb"
    ]

    for message in test_messages:

        if contains_offensive_language(message):
            print("Offensive input:", message)
            print("Bot:", get_filter_response())

        else:
            print("Safe input:", message)