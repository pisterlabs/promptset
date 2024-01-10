import math
import os
import random
import time
from datetime import date
from datetime import datetime
import openai
import requests
import speech_recognition as rec
from AppOpener import close
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from res.API_Hidden_Key import api_key
from res.Control import *
from res.simplelist import listfromtxt, txtfromlist

try:
    from spotify import *
except ImportError:
    pass

engine = pyttsx3.init()
r = rec.Recognizer()
mic = rec.Microphone()


def search_youtube(text):
    try:
        browser = webdriver.Chrome()
        browser.get("https://www.youtube.com/")
        search_box = browser.find_element(By.NAME, "search_query")
        search_box.send_keys(text)
        search_box.send_keys(Keys.RETURN)
        time.sleep(10)
        results = browser.find_elements(By.ID, "video-title")
        results[0].click()
        time.sleep(10)
        browser.quit()
    except WebDriverException:
        print("An exception occurred while using WebDriver.")
    except rec.UnknownValueError:
        print("Speech Recognition could not understand audio.")
    except rec.RequestError as e:
        print(f"Could not request results from Speech Recognition service; {e}.")
    except Exception as e:
        print(f"An unexpected error occurred; {e}.")


def say(text):
    engine.say(text)
    engine.runAndWait()


def close_app(app_name, speech_text):
    close(app_name)
    bc.say(speech_text)
    engine.runAndWait()


def open_app(app_name, speech_text):
    open(app_name)
    bc.say(speech_text)
    engine.runAndWait()


def weather(place="Chennai"):
    try:
        cite = place
        url = "https://www.google.com/search?q=" + "weather" + cite
        html = requests.get(url).content

        # getting raw data
        soup = BeautifulSoup(html, "html.parser")
        temp = soup.find("div", attrs={"class": "BNeawe iBp4i AP7Wnd"}).text
        str = soup.find("div", attrs={"class": "BNeawe tAd8D AP7Wnd"}).text

        # formatting data
        data = str.split("\n")
        sec = data[0]
        sky = data[1]
        list_div = soup.findAll("div", attrs={"class": "BNeawe s3v9rd AP7Wnd"})
        std = list_div[5].text
        pos = std.find("Wind")
        other_data = std[pos:]
        print("Temperature is", temp)
        print("Time: ", sec)
        print("Sky Description: ", sky)
        print(other_data)

        bc.say(
            f"The weather in {place} is {sky} and the temperature is {temp} and {other_data}"
        )

    except requests.exceptions.RequestException:
        print(
            "Sorry, I could not retrieve the weather data. Please check your network connection or try again later."
        )
        engine.say(
            "Sorry, I could not retrieve the weather data. Please check your network connection or try again later."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}.")
        engine.say("An unexpected error occurred.")


# def play_song():
#     with mic as origin:
#         print("Say the name of the song.")
#         say("Say the name of the song.")
#         r.adjust_for_ambient_noise(origin, duration=0.2)
#         sound = r.listen(origin)
#         print("Recognizing...")
#         say("Recognizing...")
#     try:
#         users_text = r.recognize_google(sound)
#         print("You said: " + users_text)
#     except rec.UnknownValueError:
#         print("Speech Recognition could not understand audio.")
#         say("Speech Recognition could not understand audio.")
#         return
#     except rec.RequestError as e:
#         print(f"Could not request results from Speech Recognition service; {e}.")
#         say("Could not request results from Speech Recognition service.")
#         return
#     except Exception as e:
#         print(f"An unexpected error occurred; {e}.")
#         say("An unexpected error occurred.")
#         return
#
#     song(users_text)
#     engine.say("playing music")
#     engine.runAndWait()


math_operations = {
    "addition": lambda num1, num2: num1 + num2,
    "subtraction": lambda num1, num2: num1 - num2,
    "multiplication": lambda num1, num2: num1 * num2,
    "division": lambda num1, num2: num1 / num2,
    "modulus": lambda num1, num2: num1 % num2,
    "exponential": lambda num1, num2: num1**num2,
    "square root": lambda num1: num1**0.5,
    "cube root": lambda num1: num1**0.333,
    "square": lambda num1: num1**2,
    "cube": lambda num1: num1**3,
    "log": lambda mum1: math.log(mum1),
    "log base 10": lambda mum1, num2=None: math.log10(mum1)
    if num2 is None
    else "Invalid Operation",
    "log base 2": lambda num1, num2=None: math.log2(num1)
    if num2 is None
    else "Invalid Operation",
    "factorial": lambda num1, mum2=None: math.factorial(num1)
    if mum2 is None
    else "Invalid Operation",
    "sin": lambda num1, mum2=None: math.sin(num1)
    if mum2 is None
    else "Invalid Operation",
    "cos": lambda num1, num2=None: math.cos(num1)
    if num2 is None
    else "Invalid Operation",
    "tan": lambda num1, num2=None: math.tan(num1)
    if num2 is None
    else "Invalid Operation",
    "sin inverse": lambda num1, num2=None: math.asin(num1)
    if num2 is None
    else "Invalid Operation",
    "cos inverse": lambda num1, num2=None: math.acos(num1)
    if num2 is None
    else "Invalid Operation",
    "tan inverse": lambda num1, num2=None: math.atan(num1)
    if num2 is None
    else "Invalid Operation",
    "arcs in": lambda num1, num2=None: math.asin(num1)
    if num2 is None
    else "Invalid Operation",
    "arc cos": lambda num1, num2=None: math.acos(num1)
    if num2 is None
    else "Invalid Operation",
    "arc tan": lambda num1, num2=None: math.atan(num1)
    if num2 is None
    else "Invalid Operation",
    "sinh": lambda num1, num2=None: math.sinh(num1)
    if num2 is None
    else "Invalid Operation",
    "cosh": lambda num1, num2=None: math.cosh(num1)
    if num2 is None
    else "Invalid Operation",
    "tanh": lambda num1, num2=None: math.tanh(num1)
    if num2 is None
    else "Invalid Operation",
    "sinh inverse": lambda num1, num2=None: math.asinh(num1)
    if num2 is None
    else "Invalid Operation",
    "cosh inverse": lambda num1, num2=None: math.acosh(num1)
    if num2 is None
    else "Invalid Operation",
    "tanh inverse": lambda num1, num2=None: math.atanh(num1)
    if num2 is None
    else "Invalid Operation",
    "arc sinh": lambda num1, num2=None: math.asinh(num1)
    if num2 is None
    else "Invalid Operation",
    "arc cosh": lambda num1, num2=None: math.acosh(num1)
    if num2 is None
    else "Invalid Operation",
    "arc tanh": lambda num1, num2=None: math.atanh(num1)
    if num2 is None
    else "Invalid Operation",
    "degrees": lambda num1, num2=None: math.degrees(num1)
    if num2 is None
    else "Invalid Operation",
    "radians": lambda num1, num2=None: math.radians(num1)
    if num2 is None
    else "Invalid Operation",
    "pi": lambda: math.pi,
    "e": lambda: math.e,
    "tau": lambda: math.tau,
    "gamma": lambda: math.gamma,
}

commands = {
    "open Google": lambda: open_app("google chrome", "opening google"),
    "open chrome": lambda: open_app("google chrome", "opening google"),
    "open YouTube": lambda: open_app("youtube", "opening youtube"),
    "open WhatsApp": lambda: open_app("whatsapp", "opening whatsapp"),
    "open Firefox": lambda: open_app("firefox", "opening firefox"),
    # "play a song": play_song,
    "increase brightness": increase_,
    "increase the brightness": increase_,
    "decrease volume": decrease,
    "decrease the volume": decrease,
    "increase volume": increase,
    "increase the volume": increase,
    "decrease brightness": decrease_,
    "decrease the brightness": decrease_,
    "open camera": lambda: open_app("camera", "opening camera"),
    "open calculator": lambda: open_app("calculator", "opening calculator"),
    "open Notepad": lambda: open_app("notepad", "opening notepad"),
}


def user_input():
    user_in = input("{}: ".format(user_name))
    user_in.lower()


print(
    """\033[34m
             ███████╗░█████╗░██████╗░██╗░░░██╗  ░█████╗░██╗
             ╚════██║██╔══██╗██╔══██╗╚██╗░██╔╝  ██╔══██╗██║
             ░░███╔═╝███████║██████╔╝░╚████╔╝░  ███████║██║
             ██╔══╝░░██╔══██║██╔═══╝░░░╚██╔╝░░  ██╔══██║██║
             ███████╗██║░░██║██║░░░░░░░░██║░░░  ██║░░██║██║
             ╚══════╝╚═╝░░╚═╝╚═╝░░░░░░░░╚═╝░░░  ╚═╝░░╚═╝╚═╝\033[0m"""
)
openai.api_key = api_key()

conversation = ""
bot_name = "\033[35mZapy\033[0m"
bot = "Zapy"
version = "1.0.1"
# resource initialisation
t = time.localtime()
d = date.today()
dt = datetime.now().strftime("%A")
current_time = time.strftime("%H:%M:%S", t)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (HTML, like Gecko) "
    "Chrome/58.0.3029.110 Safari/537.3"
}
greetings = ["Hello!", "What's up?!", "Howdy!", "Greetings!"]
goodbyes = ["Bye!", "Goodbye!", "See you later!", "See you soon!"]
special = ["day", "time", "weather", "/voice", "maths"]

res = [
    dt,
    current_time,
]

keywords = listfromtxt("res/data/keywords.txt")

response = listfromtxt("res/data/responses.txt")

current_time = time.strftime("%H:%M:%S", t)

initial = """
█ █▄░█ █ ▀█▀ █ ▄▀█ █░░ █ ▀█ █ █▄░█ █▀▀   ▀█ ▄▀█ █▀█ █▄█
█ █░▀█ █ ░█░ █ █▀█ █▄▄ █ █▄ █ █░▀█ █▄█   █▄ █▀█ █▀▀ ░█░"""
dot = "   ▄ ▄ ▄"
bc.say("Initialising {}".format(bot))
print(initial, end="")
for char in dot:
    print(char, end="", flush=True)
    time.sleep(0.3)
print()
print("""𝕡𝕝𝕖𝕒𝕤𝕖 𝕨𝕒𝕚𝕥...""")
say(" please wait")
time.sleep(1)
Keyword_file = os.path.exists("res/data/keywords.txt")
response_file = os.path.exists("res/data/responses.txt")
time.sleep(1)

if not Keyword_file:
    print("keyword file not found")
    time.sleep(2)
    print("creating (keywords.txt) file")
    open("res/data/keywords.txt", "x")
    print("done")
if not response_file:
    print("response file not found")
    time.sleep(2)
    print("creating (responses.txt) file")
    open("res/data/responses.txt", "x")
    print("done")

# bot starts
bc.say("AI Chat Me BOT {}, version {}, Initiated".format(bot, version))
print(
    "\033[32m{}: AI Chat Me BOT {}, versio8"
    "n {}, Initiated\033[0m".format(bot_name, bot_name, version)
)
time.sleep(1)
engine.runAndWait()
user = (
    input(
        "\033[36m select mode: \n 1. user mode\033[0m \n \033[31m2. dev mode\033[0m \n You: "
    )
    or "1"
)
save_response = "n"
if user == "1":
    print("{}: user mode selected".format(bot_name))
    say("user mode selected")
    print(
        "{}: Text to speech enabled, use /voice to enable voice mode".format(bot_name)
    )
    say("Text to speech enabled, use /voice to enable voice mode")
elif user == "2":
    print("\033[31m2{}: dev mode selected".format(bot_name))
    say("dev mode selected")
    say("Do you want to save new response?")
    save_response = input(
        "Do you want to save this response? (y/n) \n [tip: save when you are training the bot] \n You: "
    )
    print(
        "{}: Text to speech enabled, use /voice to enable voice mode".format(bot_name)
    )
    say("Text to speech enabled, use /voice to enable voice mode")

print("{}: Let me know your name, before we start".format(bot_name))
say("Let me know your name, before we start")
user_name = input("\033[35m please enter your name\033[0m:") or "user"
user_name = "{}".format(user_name)
print(
    "{}: "
    "Hello! {} I am {}, a virtual assistant."
    " I'm here to answer your questions. How can I assist you today?".format(
        bot_name, user_name, bot_name
    )
)
say(
    "Hello! {} I am {}, a virtual assistant. I am here to answer your questions. How can I assist you today?".format(
        user_name, bot
    )
)
user_name = "\033[95m{}\033[0m".format(user_name)
user = input("{}: ".format(user_name))
user = user.lower()

while user != "bye":
    keyword_found = False
    # analise
    for index in range(len(keywords)):
        if keywords[index] == user:
            print("{}: ".format(bot_name) + response[index])
            say(response[index])
            keyword_found = True
            break
    # special
    if not keyword_found:
        for index in range(len(special)):
            if special[2] in user:
                try:
                    city = input("{}: Enter the Name of City ->  ".format(bot_name))
                    city = city + " weather"
                    weather(city)
                    keyword_found = True
                except requests.exceptions.RequestException:
                    print(
                        "Sorry, I could not retrieve the weather data. Please check your network connection or try "
                        "again later."
                    )
                    say(
                        "Sorry, I could not retrieve the weather data. Please check your network connection or try "
                        "again later."
                    )
                    keyword_found = True

                break
            elif (
                user == "play a song"
                or user == "play song"
                or user == "play a song for me"
                or user == "play song " "for me"
            ):
                say("playing a song")
                song()
                keyword_found = True
                break
            elif special[4] == user:
                operation = input(
                    "Enter operation (addition, subtraction, etc, exit): "
                )
                if operation == "exit":
                    keyword_found = True
                    break

                else:

                    n1 = float(input("Enter first number: "))

                    if operation in [
                        "square root",
                        "cube root",
                        "square",
                        "cube",
                        "log",
                        "log base 10",
                        "log base 2",
                        "factorial",
                        "sin",
                        "cos",
                        "tan",
                        "sin inverse",
                        "cos inverse",
                        "tan inverse",
                        "arc sin",
                        "arc cos",
                        "arc tan",
                        "sinh",
                        "cosh",
                        "tanh",
                        "sinh inverse",
                        "cosh inverse",
                        "tanh inverse",
                        "arc sinh",
                        "arc cosh",
                        "arc tanh",
                        "degrees",
                        "radians",
                    ]:

                        # If operation requires only one input, set num2 to a default value of 0

                        n2 = None

                    else:

                        n2 = float(input("Enter second number: "))

                    result = math_operations.get(
                        operation, lambda x, y: "Invalid Operation"
                    )(n1, n2)

                    if isinstance(result, float):
                        print("Result: ", result)
                    else:
                        print(result)
                    keyword_found = True

            elif user == "hi" or user == "hello" or user == "hey":
                greet = random.choice(greetings)
                print("{}: ".format(bot_name), greet)
                say(greet)
                keyword_found = True
                break

            elif user == "change your name":
                bot_name = input("Enter new name: ")
                keyword_found = True
                break

            elif special[3] == user:
                print("voice mode enabled")
                say("voice mode enabled")
                while True:
                    with mic as source:
                        print("\033[32mListening...\033[0m")
                        say("listening...")
                        r.adjust_for_ambient_noise(source, duration=0.5)
                        audio = r.listen(source)
                        print("\033[34mRecognising...\033[0m")
                        say("recognising...")
                    try:
                        user_input = r.recognize_google(audio)
                        print("You said: " + user_input)
                    except rec.UnknownValueError:
                        print(
                            "{} :\033[31mSorry, I could not understand. Please try again.\033[0m".format(
                                bot_name
                            )
                        )
                        say("Sorry, I could not understand. Please try again.")
                        continue
                    if user_input in commands:
                        commands[user_input]()
                    elif "open YouTube and search " in user_input:
                        # Extract the query from the command
                        query = user_input.split("search ")[-1]
                        say("Searching for " + query)
                        search_youtube(query)
                    else:
                        try:
                            prompt = (
                                user_name + ": " + user_input + "\n{}:".format(bot_name)
                            )
                            conversation += prompt
                            start_time = time.time()

                            _response = openai.Completion.create(
                                engine="text-davinci-003",
                                prompt=conversation,
                                max_tokens=50,
                            )

                            response_string = _response["choices"][0]["text"].replace(
                                "\n", " "
                            )
                            response_string = response_string.split(user_name + ":", 1)[
                                0
                            ].split("{}:".format(bot_name), 1)[0]
                            elapsed_time = time.time() - start_time
                            if elapsed_time > 10:
                                print("Please wait, I'm thinking...")
                                say("Please wait, I'm thinking...")
                                continue

                            conversation += response_string + "\n"

                            print("{}: ".format(bot_name) + response_string)
                            say(response_string)
                        except requests.exceptions.RequestException:
                            print("Check your connection")
                            say("check your internet connection")
                            continue

                    if user_input == "exit":
                        break
                keyword_found = True
                break
    if not keyword_found:
        user_input = user

        try:
            prompt = user_name + ": " + user_input + "\n{}:".format(bot_name)
            conversation += prompt
            start_time = time.time()

            _response = openai.Completion.create(
                engine="text-davinci-003", prompt=conversation, max_tokens=50
            )

            response_string = _response["choices"][0]["text"].replace("\n", " ")
            response_string = response_string.split(user_name + ":", 1)[0].split(
                "{}:".format(bot_name), 1
            )[0]
            elapsed_time = time.time() - start_time
            if elapsed_time > 10:
                print("Waiting...")
                say("Please wait, I'm thinking...")
                continue

            conversation += response_string + "\n"

            print("{}: ".format(bot_name) + response_string)
            say(response_string)
        except:
            print("Check your connection")
            new_keyword = user
            keywords.append(new_keyword)
            print("{}: SORRY!! I'm not sure how to respond".format(bot_name))
            print("{}: How should I respond to [ {} ] ?".format(bot_name, new_keyword))
            new_response = input("BotResponse: ")
            if new_response == "exit":
                break
            else:
                response.append(new_response)
                txtfromlist("res/data/keywords.txt", keywords)
                txtfromlist("res/data/responses.txt", response)
                print("{}: New response has been updated".format(bot_name))
            keyword_found = True
        if save_response == "y":
            keywords.append(user)
            txtfromlist("res/data/keywords.txt", keywords)
            txtfromlist("res/data/responses.txt", response)
            print("Response saved")
            say("Response saved")

    user = input("{}: ".format(user_name))
    user = user.lower()

print("{}:".format(bot_name), random.choice(goodbyes))
