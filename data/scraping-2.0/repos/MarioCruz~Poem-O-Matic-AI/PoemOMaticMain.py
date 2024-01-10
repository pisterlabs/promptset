import openai
import os
import time
import pynput.keyboard
from google.cloud import texttospeech
from concurrent.futures import ThreadPoolExecutor
import threading
from pynput.keyboard import Key, Listener


# Set API keys and environment variables
openai.api_key = os.environ.get('OPENAI_API_KEY')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "poem-0-matic-b9f292c4f549.json"

stop_reading_choices = threading.Event()
user_input = None

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_separator():
    print("-" * 50)

def on_key_release(key):
    global user_input
    try:
        user_input = key.char
    except AttributeError:
        user_input = None
    if user_input in valid_keys:
        return False

def speak_text_google(text):
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
            name="en-US-Wavenet-F"
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        with open("output.mp3", "wb") as out:
            out.write(response.audio_content)
        os.system("afplay output.mp3")  # for MacOS
        os.remove("output.mp3")
    except Exception as e:
        print(f"\nOops! Something went wrong with the text-to-speech functionality. Here's what happened:\n{e}")
        print("Please check your Google TTS configuration or ensure you're online. Try again later.")
        print_separator()

def read_choices_out_loud(options):
    for i, option in enumerate(options, 1):
        if stop_reading_choices.is_set():
            break
        print(f"{i}. {option}")
        speak_text_google(f"Option {i}. {option}")

#_________________________ fixing this 

def get_user_choice(options, prompt):
    choice = None
    stop_reading_choices = threading.Event()

    while True:  # Keep asking until a valid choice is made
        print(prompt)
        choice_input_message = "\nPress the corresponding number for your choice as soon as you decide and let the magic unfold!\n"
        print(choice_input_message, end='', flush=True)

        valid_keys = [str(i) for i in range(1, len(options) + 1)]

        def on_key_release(key):
            nonlocal choice
            try:
                user_input = key.char
                if user_input in valid_keys:
                    choice = int(user_input)
                    stop_reading_choices.set()
                    return False
                else:  # Handle invalid input immediately
                    print("\nInvalid choice. Please select a valid number from the list.")
            except AttributeError:
                pass

        read_thread = threading.Thread(target=read_choices_out_loud, args=(options,))
        read_thread.start()

        with pynput.keyboard.Listener(on_release=on_key_release) as listener:
            listener.join()

        read_thread.join()

        if choice:
            return options[choice-1]
        else:
            print("Hmm, that choice doesn't seem right. Please pick a number from the list.")
            print_separator()
            # There's no need for a recursive call or a continue statement. The loop will naturally continue.


#_________________________ fixing this 


def fetch_poem_from_openai(prompt):
    retries = 3
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                                    messages=[{"role": "system", "content": "You are an awesome poet that grew up in miami, florida"},
                                                              {"role": "user", "content": prompt}])
            content = response.choices[0]['message']['content']
            if content:
                return content.strip()
            else:
                return "Error: The response from OpenAI was unexpected."
        except openai.error.OpenAIError as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            else:
                print(f"\nThere seems to be an issue with fetching the poem from OpenAI after {retries} attempts. Here's the technical detail:\n{e}")
                print("Ensure your API key is set correctly and that you have network access. Maybe try again in a moment.")
                print_separator()
        except Exception as e:
            print(f"\nUnexpected error while generating a poem. Here's what happened:\n{e}")
            print("Maybe try restarting the application or checking for updates.")
            print_separator()

def get_user_inputs():
    colors = ["Red", "Yellow", "Blue", "Orange", "Green"]
    locations = ["Miami Beach", "Downtown Miami", "Wynwood Miami", "Underline Miami", "Little Havana, Miami"]
    feelings = ["Happy", "Anxious", "Nostalgic", "Frustrated", "Romantic"]
    wildcards = ["Mojito", "Music", "Magic", "Epic", "Books"]
    poem_styles = ["Sonnet", "Haiku", "Free Verse", "Limerick", "Villanelle"]

    color_choice = get_user_choice(colors, "Choose a color:")
    location_choice = get_user_choice(locations, "Choose a location:")
    feeling_choice = get_user_choice(feelings, "Choose a feeling:")
    wildcard_choice = get_user_choice(wildcards, "Choose a wildcard:")
    poem_style_choice = get_user_choice(poem_styles, "Choose a poem style:")

    return color_choice, location_choice, feeling_choice, wildcard_choice, poem_style_choice

def generate_poem_with_openai():
    color_choice, location_choice, feeling_choice, wildcard_choice, poem_style_choice = get_user_inputs()
    clear_screen()
    
    prompt = (f"Write a {poem_style_choice} about {location_choice} with themes of {color_choice}, "
             f"{feeling_choice}, and {wildcard_choice}.")

    with ThreadPoolExecutor() as executor:
        future_poem = executor.submit(fetch_poem_from_openai, prompt)

        choices = {
            "Color": color_choice,
            "Location": location_choice,
            "Feeling": feeling_choice,
            "Wildcard": wildcard_choice,
            "Poem Style": poem_style_choice
        }

        display_choices(choices)
        poem = future_poem.result()
        return poem

def display_choices(choices):
    print("\nBased on your choices, here's the theme for your personalized poem:")
    print_separator()
    for key, value in choices.items():
        print(f"{key}: {value}")
    print_separator()

if __name__ == '__main__':
    clear_screen()
    print_separator()
    print("Welcome to Poem-O-Matic-AI by MarioTheMaker")
    print_separator()
    print("\nLet's craft you a beautiful poem! Please make your choices...")
    print_separator()
    poem = generate_poem_with_openai()
    print(f"\nAnd here's your poem:\n\n{poem}\n")
    print_separator()
    speak_text_google(poem)
    time.sleep(2)
    sign_off = "Poem Created by Poem-O-Matic-AI by Mario The Maker for O'Miami festival"
    print(sign_off)
    print_separator()
    speak_text_google(sign_off)
