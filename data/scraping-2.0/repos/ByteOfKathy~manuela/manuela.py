import random
from nltk.parse.dependencygraph import malt_demo
import text2emotion as te
import textRecognizer as tr
import numpy as np
import moodDetection as md
import openaiIntegration as oi


def get_emotion_helper() -> dict:
    """
    Gets the emotion from the user
    :return: the emotion from the user
    """
    tr.tts("Hello! I'm Manuela. What's your name? ")
    name = tr.recognizeSpeech()
    tr.tts("Hello " + name + ", how are you feeling today?")

    talk_to_manuela(name)


def talk_to_manuela(name: str):
    """
    Responds to the emotion
    :param emotion: the emotion to respond to
    :param name: the name of the user
    """

    # Prompt from the user
    userInput = tr.recognizeSpeech()
    tr.tts(
        "I see, give me some time to process how you are feeling, it must be a lot for you as much as it is for me."
    )
    emotion_weights: dict = te.get_emotion(userInput)
    emotion_weights["neutral"] = 0
    emotion = md.interpret_emotion(emotion_weights).lower()

    tr.tts(
        f"From what I can see and hear, it seems you're mostly feeling {emotion}. Is that correct? A good old yes or no will do the trick."
    )

    correct_response = tr.recognizeSpeech()
    if correct_response.lower() == "yes":
        if len(md.available_emotions) == 1 and emotion == "neutral":
            tr.tts(
                oi.responseGenerator(emotion, userInput)
            )  # To change later to some custom respone for not understanding the user's emotion
        elif emotion in md.available_emotions:
            tr.tts(
                f"Well, {name} give me some time to think, I will be right back with some suggestions."
            )
            tr.tts(oi.responseGenerator(emotion, userInput))
        else:
            tr.tts(
                "I'm sorry, I don't understand that emotion. Could you explain further?"
            )
            talk_to_manuela()
    else:
        tr.tts("I'm sorry, I must have misunderstood. Could you tell me more?")
        if emotion != "neutral":
            md.available_emotions.remove(emotion)
        talk_to_manuela()


if __name__ == "__main__":
    get_emotion_helper()
