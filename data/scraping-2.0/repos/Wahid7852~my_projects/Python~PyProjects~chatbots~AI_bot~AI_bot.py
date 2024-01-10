import speech_recognition as sr
from datetime import date
import pyttsx3
import datetime
import os
import pyjokes
import openai

# Set up OpenAI API key
openai.api_key = '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'

engine = pyttsx3.init()

COMMANDS = {
    'time': lambda: get_current_time(),
    'date': lambda: get_current_date(),
    'how are you': lambda: respond('I am absolutely fine, thank you'),
    'exit': lambda: exit_program(),
    'introduce': lambda: introduce_bot(),
    'help': lambda: show_help(),
    'joke': lambda: tell_joke(),
}


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def get_greeting():
    hour = datetime.datetime.now().hour
    if 0 <= hour < 12:
        return "Good Morning"
    elif 12 <= hour < 18:
        return "Good Afternoon"
    else:
        return "Good Evening"


def respond(response_text):
    print(response_text)
    speak(response_text)


def get_current_time():
    current_time = datetime.datetime.now().strftime('%I:%M %p')
    respond(f"Right now, the time is {current_time}")


def get_current_date():
    current_date = date.today().strftime('%d %B %Y %A')
    respond(f"Today, it is {current_date}")


def exit_program():
    respond('Thank you for using me, have a nice day')
    exit()


def introduce_bot():
    introduction = "Hello! I'm VoiceBot, your personal assistant. I'm here to make your life easier. You can ask me about the time, date, or even tell me how you're feeling. If you're not sure what to do, just say 'help' for more options.\n"
    respond(introduction)


def show_help():
    help_message = "Here are some things you can do with VoiceBot:\n" \
                   "- Ask for the time\n" \
                   "- Ask for the date\n" \
                   "- Ask for help if you're not sure what to do\n"
    respond(help_message)

def choose_input_method():
    while True:
        choice = input("Choose input method (mic/keyboard): ").lower()
        if choice in ('mic', 'keyboard'):
            return choice
        else:
            print("Invalid choice. Please enter 'mic' or 'keyboard'.")

def tell_joke():
    joke = pyjokes.get_joke()
    respond(joke)


def process_query(query):
    for command, function in COMMANDS.items():
        if command in query:
            function()
            return True

    if 'joke' in query:
        tell_joke()
        return True
    return False


def generate_openai_response(query):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=query,
        max_tokens=50,
    )
    return response.choices[0].text


def take_command(input_method):
    if input_method == "mic":
        try:
            with sr.Microphone() as source:
                r = sr.Recognizer()
                r.energy_threshold = 10000
                r.adjust_for_ambient_noise(source, 1)
                print('Listening...')
                audio = r.listen(source)

            print('Recognizing...')
            query = r.recognize_google(audio, language='en')
            print(f'User said: {query}\n')
            return query.lower()

        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Please say it again.")
        except sr.RequestError:
            print(
                "I'm sorry, I couldn't request the recognition service. Please check your internet connection.")

    elif input_method == "keyboard":
        query = input("Your command: ")
        return query.lower()


def main():
    os.system('cls')
    get_greeting()
    introduce_bot()
    input_method = choose_input_method()

    while True:
        query = take_command(input_method)
        if not process_query(query):
            openai_response = generate_openai_response(query)
            respond(openai_response)


if __name__ == "__main__":
    main()
