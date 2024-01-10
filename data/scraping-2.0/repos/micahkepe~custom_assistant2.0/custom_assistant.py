import speech_recognition as sr
import openai
import PySimpleGUI as sg
from elevenlabslib import *
from pathlib import Path
from configparser import ConfigParser
import requests
from dotenv import load_dotenv
import os
import logging
import json
import time

# Load OpenAI and ElevenLabs keys from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_KEY = os.getenv("ELEVENLABS_KEY")

# find microphone to use later to record audio
r = sr.Recognizer()
mic = sr.Microphone()

# model cache
CACHE_FILE = "models_cache.json"
CACHE_TIME = 60 * 60  # 1 hour


def get_users_models():
    """
    Get a list of the available voices in the user's ElevenLabs account
    :return: list of strings of available voice models
    """
    models = []

    # check if cache file exists
    if os.path.exists(CACHE_FILE):
        # check if cache file is older than CACHE_TIME
        if time.time() - os.path.getmtime(CACHE_FILE) > CACHE_TIME:
            # delete cache file
            logging.info(
                f"Cache file is older than {CACHE_TIME} seconds. Deleting cache file."
            )
            os.remove(CACHE_FILE)
        else:
            # read cache file
            logging.info(
                f"Cache file is newer than {CACHE_TIME} seconds. Reading cache file."
            )
            with open(CACHE_FILE, "r") as f:
                models = json.load(f)
            return models

    # HTTP request to retrieve the voices
    API_KEY = ELEVENLABS_KEY
    BASE_URL = "https://api.elevenlabs.io/v1"
    ENDPOINT = "/voices"
    headers = {"accept": "application/json", "xi-api-key": API_KEY}

    response = requests.get(BASE_URL + ENDPOINT, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        voices = response.json()["voices"]
        voice_names = [voice["name"] for voice in voices]
        for name in voice_names:
            models.append(name)
    else:
        sg.popup_error(
            "ElevenLabs Key is not valid. Try different key at top of program."
        )
        logging.error(
            f"ElevenLabs Key is not valid. HTTP Response Code: {response.status_code}"
        )

    # write cache file
    logging.info(f"Writing cache file.")
    with open(CACHE_FILE, "w") as f:
        json.dump(models, f)

    return models


def has_selected_model(user_values):
    """
    Checks if user has selected a model
    :param user_values: dictionary of values associated with user's window
    :return: None
    """
    if user_values["-Selected Model-"] == "":
        sg.popup("Please select a model.")
        return False

    return True


def has_name(user_values):
    """
    Checks if user has inputted a name
    :param user_values: dictionary of values associated with user's window
    :return: None
    """
    if user_values["-USER NAME-"] == "":
        sg.popup("Please write your name.")
        return False

    return True


def is_valid_user():
    """ "
    Checks if user's keys are valid
    :return: boolean value of valid or not
    """
    try:
        ElevenLabsUser(ELEVENLABS_KEY)
        openai.api_key = OPENAI_API_KEY
        openai.Completion.create(engine="text-davinci-003", prompt="Test", max_tokens=5)
        return True

    except Exception as e:
        sg.popup_error(
            "OpenAI and/or ElevenLabs Key is not valid. Try different key(s) at top of program."
        )
        logging.error(f"OpenAI and/or ElevenLabs Key is not valid. Exception: {e}")
        return False


def generate_image_response(prompt):
    """
    Generate response from GPT/ DallE for a query involving drawing a picture
    :param prompt: string of the user's query
    :return: string image url of generated drawing, index of rest of string after "draw"
    """
    # find subject user wants
    i = prompt.find("draw")
    i += 5

    # generate GPT response
    gpt_response = openai.Image.create(prompt=prompt[i:], n=1, size="1024x1024")

    # get url from response and return
    url = gpt_response["data"][0]["url"]
    return url, i


def generate_text_response(prompt, ongoing_convo):
    """
    Generate response from GPT for a standard query not involving drawing a picture
        and update chat log
    :param prompt: string of the user's query
    :param ongoing_convo: list of dictionaries of ongoing conversations between user and model
    :return: string of GPT's response
    """
    ongoing_convo.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=ongoing_convo,
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0.5,
    )

    answer = response["choices"][0]["message"]["content"]
    ongoing_convo.append({"role": "assistant", "content": answer})
    return answer


def generate_and_show_response(prompt, model, ongoing_convo, voice, values):
    """
    Generates response to user's question and displays answer in popups and terminal
    :param prompt: string of the user's query
    :param model: model selected by the user from available models
    :param ongoing_convo: list of dictionaries representing ongoing conversation
    :param voice: ElevenLabsVoice object tied to selected model
    :param values: dictionary of values associated with main window
    :return: None
    """
    # if user wants to assistant to draw something (has "draw" in prompt)
    if "draw" in prompt:
        image_url, idx = generate_image_response(prompt)
        print(f"{model}: Here's {prompt[idx:]}")
        print(image_url)
        sg.popup_scrolled(
            f"You said: {prompt} \n {model}: Here's {prompt[idx:]} \n {image_url}"
        )
        print("=====")

    # if user asks a standard question (no "draw" request)
    else:
        message = generate_text_response(prompt, ongoing_convo)

        # Show GPT's response
        if values["-Spoken Response-"] == "Yes":
            voice.generate_and_play_audio(message, playInBackground=False)

        print(f"{model}: {message}")
        sg.popup_scrolled(f"You said: {prompt}\n\n{model}: {message}")

        print("=====")


def main_window():
    # Cache available models and voice objects
    available_models = get_users_models()
    voices = {model: user.get_voices_by_name(model)[0] for model in available_models}

    # GUI Definition
    sg.theme("dark grey 9")
    layout = [
        [sg.Text("Your Name:"), sg.Input(key="-USER NAME-")],
        [
            sg.Text("Model:"),
            sg.Combo(available_models, readonly=True, key="-Selected Model-"),
        ],
        [
            sg.Text("Spoken Responses:"),
            sg.Combo(
                ["Yes", "No"],
                readonly=True,
                key="-Spoken Response-",
                default_value="Yes",
            ),
        ],
        [sg.Text("Query Mode:"), sg.Button("Speak"), sg.Button("Type")],
        [sg.Save("Save"), sg.Exit("Exit")],
    ]

    # Create the window
    window_title = settings["GUI"]["title"]
    window = sg.Window(window_title, layout)

    # Store ongoing conversations
    conversations = {}

    while True:
        event, values = window.read()

        # User closes window
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        # Handle events with exceptions
        try:
            if (
                event in ("Speak", "Type")
                and has_name(values)
                and has_selected_model(values)
            ):
                name = values["-USER NAME-"]
                model = values["-Selected Model-"]
                voice = voices[model]  # Reuse voice object from cache

                # Initialize conversation for this model if not already done
                if model not in conversations:
                    conversations[model] = [
                        {
                            "role": "system",
                            "content": f"Your name is {model} and you're an assistant for {name}.",
                        },
                    ]

                conversation = conversations[model]

                # Handle Speak event
                if event == "Speak":
                    with mic as source:
                        r.adjust_for_ambient_noise(
                            source, duration=1
                        )  # Can set the duration with duration keyword

                        print("Speak now...")
                        sg.popup_timed(
                            "Speak now...", auto_close=True, auto_close_duration=1
                        )

                        try:
                            # Gather audio and transcribe to text
                            audio = r.listen(source)
                            word = r.recognize_google(audio)

                            # Show user's query
                            print(f"You said: {word}")

                            # Close window and quit program when user says "That is all"
                            if word.lower() == "that is all":
                                print(f"{model}: See you later!")
                                sg.popup_timed(
                                    f"{model}: See you later!",
                                    auto_close=True,
                                    auto_close_duration=2,
                                )
                                window.close()
                                quit()

                            # Generate and display response
                            generate_and_show_response(
                                word, model, conversation, voice, values
                            )

                        except Exception as e:
                            print(f"Couldn't interpret audio, try again. Error: {e}")
                            print("=====")

                # Handle Type event
                elif event == "Type":
                    # Have user type question in popup
                    word = sg.popup_get_text("Enter question")

                    # Show user's query
                    print(f"You said: {word}")

                    # Generate and display response
                    generate_and_show_response(word, model, conversation, voice, values)
        except Exception as e:
            logging.error(f"Error: {e}")
            sg.popup_error(f"Error: {e}")

        # Ensure window closes
        finally:
            window.close()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Check if the user is a valid user from the provided keys
    is_valid_user()

    user = ElevenLabsUser(ELEVENLABS_KEY)

    SETTINGS_PATH = str(Path.cwd())
    # create the settings object and use ini format
    settings = sg.UserSettings(
        path=SETTINGS_PATH,
        filename="config.ini",
        use_config_file=True,
        convert_bools_and_none=True,
    )
    configur = ConfigParser()
    configur.read("config.ini")

    theme = configur.get("GUI", "theme")
    font_family = configur.get("GUI", "font_family")
    font_size = configur.getint("GUI", "font_size")
    sg.theme(theme)
    sg.set_options(font=(font_family, font_size))

    main_window()
