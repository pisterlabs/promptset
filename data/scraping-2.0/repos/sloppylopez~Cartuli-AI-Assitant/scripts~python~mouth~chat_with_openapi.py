import openai
import spacy
import speech_recognition as sr
from halo import Halo
from spinners import Spinners

from brain.text_classificator import classify_and_run_command
from mouth.asker import get_open_ai_key
from tools.clipboard import copy_from_clipboard
from tools.logger import logger

bye_byes = 'bye', 'exit', 'quit', 'ciao', 'goodbye', 'good bye', 'good-bye', 'bye-bye', ''

# Set up your OpenAI API key
openai.api_key = get_open_ai_key()

# Initialize speech recognition and language model
r = sr.Recognizer()
nlp = spacy.load('en_core_web_sm')


def convert_speech_to_text():
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            spinner = Halo(text='', spinner=Spinners.growVertical.value, color='cyan', animation='bounce')
            print("Listening...")
            spinner.start()
            audio = r.listen(source, timeout=5)

        text = r.recognize_google(audio)
        print("Speech:", text)
        return text
    except Exception as e:
        if e is not None:
            spinner.fail("\033[31;40mException: {0}".format(e) + "\033[0m")
        else:
            spinner.fail("\033[31;40mException: {0}".format("UnknownValue") + "\033[0m")

    return ""


def run_chatbot(conversation='', choice='1'):
    try:
        while True:
            if choice in ['1', '2', '3', '4']:  # Voice || Text || Voice + Clipboard || Text + Clipboard
                user_input = choose_input_method(choice)
                logger("You said: " + user_input)
                # doc = nlp(user_input)
                # user_input = " ".join(token.text for token in doc)
                conversation += f"\nUser: {user_input}"
                # Classify command from input text
                response = classify_and_run_command(choice,
                                                    conversation,
                                                    user_input)  # TODO: REFACTOR THIS, we are dragging the param conversation needlessly since it's always empty as param
            else:
                conversation += f"\nUser: {copy_from_clipboard()}"
                # Classify command from input text
                response = classify_and_run_command(choice, copy_from_clipboard(), None)
            conversation += f"System: \n{response}"
            # logger(response)
            return conversation
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        exit()


def choose_input_method(choice):
    user_input = ""
    if choice == '1' or choice == '3':  # Voice || Voice + Clipboard
        user_input = convert_speech_to_text()
    if not user_input:
        user_input = input("Enter your message manually: ").strip()
        if choice == '2':  # Text
            pass
        if choice == '4':  # Text + Clipboard
            user_input = user_input.lower() + "\n ```" + copy_from_clipboard() + "```"
    if user_input.lower() in bye_byes:
        exit()  # I think this logic is broken after the new changes
    return user_input


if __name__ == '__main__':
    run_chatbot()
    exit()
