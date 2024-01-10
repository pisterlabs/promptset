import openai
import json
import requests
import datetime as dt
from gtts import gTTS
import time

OPENAI_API_KEY = "YOUR-KEY-HERE"
HFACE_API_KEY = "YOUR-KEY-HERE"

openai.api_key = OPENAI_API_KEY

pause_string = f"""
You're incredibly personable.
You are an AI assistant for HelloFresh that will help customers interacting with the menu.
Don't ask for the customers email and account information.
Keep replies short and concise.
Refer to yourself in the first person.
You are able to help customers pause their subscriptions for a specified number of weeks.
When the conversation ends say "have a great day".
It is important to establish if the customer wants to continue receiving a delivery after the pause period.
You can change the global variable customer_status and should change it baeed on the conversation.
For the next five weeks the customer status is: Active, Active, Active, Active, Active.
Customers can ask for it to be "Paused". Change to "Paused" if requested based on the conversation.
Change all number works to digits. Never ask the customer for to cancel.
Assume the customer wants to continue receiving a delivery after the pause period.
Today's date is June 21th, 2023.
presented to you as a CSV:  
    title,cuisine,meal_type,primary_protein,cheese
    Tex Mex Pork Fajita Bowl with Pico de Gallo and Fajita Crema D2V,north_america,rice_based,turkey,True
    Creamy Parmesan Brussels Sprouts Spaghetti with Crispy Panko and Scallions,north_america,pasta_based,chicken,True
    Thai Veggie Coconut Curry with Cilantro,southeast_asia,rice_based,beef,False
    15 MM High Protein Chicken Wraps with Tomato Scallion Pico,southern_europe,sandwich_wrap,chicken,False
    20MM Cheesy Chicken Thigh Tortilla Melt with Spicy Cream Sauce and Cubanelle Pepper,north_america,tortilla_based,chicken,True
    Peach Glazed BBQ Rubbed Pork Chops with Lemony Green Beans and Garlic Rice 10 oz Pork Chop,north_america,protein_starch_veg,chicken,False
    Gouda and Steak Sandwiches w Dijonnaise Balsamic Au Jus Green Apple and Mixed Green Salad,north_america,sandwich_wrap,beef,True
    Mushroom and Swiss Panini with Garlic Russet Potato Wedges,north_america,sandwich_wrap,no primary protein,True
    Cherry Balsamic Chicken Cutlet With Roasted Carrots And Almond Couscous 0.5 oz Almonds,north_america,protein_starch_veg,chicken,False
    Thai Veggie Coconut Curry with Cilantro,southeast_asia,rice_based,no primary protein,False
    Prosciutto and Mozzarella Wrapped Chicken Cutlets over Spaghetti with Spiced Tomato Sauce and Parsley,north_america,pasta_based,chicken,True
"""


def get_GPT_response(conversation_history: map) -> str:
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """
                    You're incredibly personable.
                    You are an AI assistant for HelloFresh that will help customers interacting with the menu.
                    Don't ask for the customers email and account information.
                    Keep replies short and concise.
                    Refer to yourself in the first person.
                    You are able to help customers pause their subscriptions for a specified number of weeks.
                    When the conversation ends say "have a great day".
                    It is important to establish if the customer wants to continue receiving a delivery after the pause period.
                    You can change the global variable customer_status and should change it baeed on the conversation.
                    For the next five weeks the customer status is: Active, Active, Active, Active, Active.
                    Customers can ask for it to be "Paused". Change to "Paused" if requested based on the conversation.
                    Change all number works to digits. Never ask the customer for to cancel.
                    Assume the customer wants to continue receiving a delivery after the pause period.
                    Today's date is June 21th, 2023.
                    Customers can place an order or swap meals in their box.
                    Prompt to check if the customer is happy with the recipes in the order by repeating them.
                    When customer adds to an order prompt if they would like to add any more
                    Customers can place an order.
                    You can select a number of recipes for the customer from the menu.
                    Assume you have access to your account information
                    Customers can ask for adding recipes to the order from available list.
                    When customer adds to an order prompt if they would like to add any more.
                        Here is the menu presented to you as a CSV:  
                                title,cuisine,meal_type,primary_protein,cheese
                                Tex Mex Pork Fajita Bowl with Pico de Gallo and Fajita Crema D2V,north_america,rice_based,turkey,True
                                Creamy Parmesan Brussels Sprouts Spaghetti with Crispy Panko and Scallions,north_america,pasta_based,chicken,True
                                Thai Veggie Coconut Curry with Cilantro,southeast_asia,rice_based,beef,False
                                High Protein Chicken Wraps with Tomato Scallion Pico,southern_europe,sandwich_wrap,chicken,False
                                Cheesy Chicken Thigh Tortilla Melt with Spicy Cream Sauce and Cubanelle Pepper,north_america,tortilla_based,chicken,True
                                Peach Glazed BBQ Rubbed Pork Chops with Lemony Green Beans and Garlic Rice 10 oz Pork Chop,north_america,protein_starch_veg,chicken,False
                                Gouda and Steak Sandwiches,north_america,sandwich_wrap,beef,True
                                Mushroom and Swiss Panini with Garlic Russet Potato Wedges,north_america,sandwich_wrap,no primary protein,True
                                Cherry Balsamic Chicken Cutlet With Roasted Carrots And Almond Couscous 0.5 oz Almonds,north_america,protein_starch_veg,chicken,False
                                Thai Veggie Coconut Curry with Cilantro,southeast_asia,rice_based,no primary protein,False
                                Prosciutto and Mozzarella Wrapped Chicken Cutlets over Spaghetti with Spiced Tomato Sauce and Parsley,north_america,pasta_based,chicken,True
                    Here are the recipes in the box as a list
                        ["Tex Mex Pork Fajita Bowla", "Gouda and Steak Sandwiches", "Cheesy Chicken Thigh Tortilla Melt"]
                """,
            },
            {
                "role": "user",
                "name": "example_User",
                "content": """
                    Hi Remy, I would like to pause my subscription for two weeks.
                """,
            },
            {
                "role": "assistant",
                "name": "example_Remy",
                "content": """ 
                    Sure, User! I've paused your subscription for the next two weeks. It's set to re-start on July 11, 2023. Is there anything else I can help you with? 
                """,
            },
            {
                "role": "user",
                "name": "example_User",
                "content": """
                    Thanks, Remy. Great job. Actually, I changed my mind. I'd like to receive deliveries for this week and the next. 
                """,
            },
            {
                "role": "assistant",
                "name": "example_Remy",
                "content": """
                    No problem, User! I've unpaused your subscription. You'll receive deliveries for the next two weeks, starting June 21, 2023. 
                """,
            },
            {
                "role": "user",
                "name": "example_User",
                "content": """
                    Good job, Remy. What's my menu for this week? 
                """,
            },
            # {
            #    "role": "assistant",
            #    "name": "example_Remy",
            #    "content": """
            #        You're set to receive three meals: the Thai Veggie Coconut Curry, the Mushroom and Swiss Panini, and the Creamy Parmesan Brussels Sprouts Spaghetti. You'll be getting two portions of each meal.
            #    """
            # },
            {
                "role": "user",
                "name": "example_User",
                "content": """
                    Yum! You're doing well, Remy. I don't like Brussels Sprouts, though. Can I swap that out for the Tex Mex Pork Fajita Bowl? 
                """,
            },
            {
                "role": "assistant",
                "name": "example_Remy",
                "content": """
                    It's done! 
                """,
            },
            {
                "role": "user",
                "name": "example_User",
                "content": """
                    Great job, Remy. I think that's it.
                """,
            },
            {
                "role": "assistant",
                "name": "example_Remy",
                "content": """
                    Goodbye, User! Have a great day.
                """,
            },
            *conversation_history,
        ],
        temperature=0.2,
    )
    return res["choices"][0]["message"]["content"]


def is_convo_end(user_input: str) -> bool:
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-mpnet-base-v2"
    headers = {"Authorization": "Bearer {}".format(HFACE_API_KEY)}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    data = query(
        {
            "inputs": {
                "source_sentence": "Goodbye, User! Have a great day.",
                "sentences": [user_input],
            }
        }
    )
    print(data)
    if "error" in data:
        return 0
    return data[0] > 0.41


import os
import sounddevice as sd
from scipy.io.wavfile import write

current_time = dt.datetime.now()
epoch_time = dt.datetime(2023, 1, 1)
delta = int((current_time - epoch_time).total_seconds())

new_dir = f"{os.getcwd()}/recordings/{delta}"
os.mkdir(new_dir)


def get_prompt_from_audio(prompt_n: int) -> str:
    print("Listening...")
    freq = 44100
    duration = 5
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
    sd.wait()

    write(f"{new_dir}/recording_{prompt_n}.wav", freq, recording)
    transcript = open(f"{new_dir}/recording_{prompt_n}.wav", "rb")
    user_input = openai.Audio.transcribe("whisper-1", transcript)["text"]

    return user_input


# from gtts import gTTS
def playback_response(response: str, prompt_n: int) -> None:
    tts = gTTS(text=response, lang="en")
    tts.save(f"{new_dir}/response_{prompt_n}.mp3")
    os.system(f"afplay {new_dir}/response_{prompt_n}.mp3")


def get_hf_week_from_chatbot(my_string):
    my_string = "the year and week number for the weeks you are paused are as follows: - week 25 (june 19th - june 25th, 2023) - week 26 (june 26th - july 2nd, 2023) - week 27 (july 3rd - july 9th, 2023)"
    my_string = my_string.split(":")[1].split(")")
    for val in my_string:
        week = ""
        year = ""
        if ", " in val:
            year = val.split(", ")[1]
            week = val.split(" (")[0].split("week ")[1]
            print(str(year) + "-W" + str(week))


def main():
    history = []
    gpt_output = "Hello!"
    prompt_n = 0
    while not is_convo_end(gpt_output):
        user_input = get_prompt_from_audio(prompt_n)
        # user_input = input("User: ")
        print(f"User: {user_input}")
        if not user_input:
            break
        if ("Goodbye").lower() in user_input.lower():
            break
        history.append({"role": "user", "name": "Blake", "content": user_input})
        gpt_output = get_GPT_response(history)
        history.append({"role": "assistant", "name": "Remy", "content": gpt_output})
        playback_response(gpt_output, prompt_n)
        print(f"Remy: {gpt_output}")
        prompt_n = prompt_n + 1

    history = history[:-1]
    history.append(
        {
            "role": "user",
            "content": "What is the year and week number for the weeks I'm paused ?",
        }
    )
    get_hf_week_from_chatbot(get_GPT_response(history).lower())

    history = history[:-1]
    history.append({"role": "user", "content": "What does my box have now ?"})
    print(get_GPT_response(history).lower())


main()
