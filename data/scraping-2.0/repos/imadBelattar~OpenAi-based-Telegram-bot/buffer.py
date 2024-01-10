from vars import *
import telegram.ext
import os
import openai
import requests
from gtts import gTTS
from io import BytesIO
from youtube_search import YoutubeSearch
import graphyte
import time
import threading
import psutil

GRAPHITE_HOST = "localhost"
GRAPHITE_PORT = 2003  # replace with the port used by your Graphite container
requests_counter = 0
responses_time = 0
time_period = 10
process = psutil.Process()
graphyte.init(GRAPHITE_HOST, GRAPHITE_PORT)
lpISIR_users = []


# allowing the owner of the bot to access the services without using the password
authentified_users = {1631515390: "owner"}
password = "graphite"
forHelp = 0
Token = os.environ["telegram-ChatWMe-bot_Token"]
openai.api_key = os.environ["OPENAI_API_KEY"]
print(f".......Running ..........ChatWithMe-bot......... by Imad Belattar.")


def start(update, context):
    response_start_time = time.time()
    global requests_counter
    global responses_time

    global requests_counter
    # Increment the requests counter for every message received
    requests_counter += 1
    update.message.reply_text(
        "Hi, how can I assist you ! (if you want answers in arabic, ask like this 'ar your question..')"
    )
    response_end_time = time.time()

    responses_time += response_end_time - response_start_time


def help(update, context):
    response_start_time = time.time()
    global requests_counter
    global responses_time

    global forHelp
    # Increment the requests counter for every message received
    requests_counter += 1
    forHelp = forHelp + 1
    if forHelp <= 2:
        update.message.reply_text("there is no help command ok !")
    else:
        update.message.reply_text("not again, you can do better than that")
    response_end_time = time.time()
    print(
        f"   processed response time : {response_end_time - response_start_time} seconds"
    )

    responses_time += response_end_time - response_start_time


# retreive a youtube video function


def send_video(name, chat_id, bot_token):
    # Search for the song on YouTube
    results = YoutubeSearch(name, max_results=1).to_dict()

    if results:
        # Get the URL of the first search result
        video_url = f"https://www.youtube.com/watch?v={results[0]['id']}"
        return f"🎥 🎶 🔊 Here's a link of {name} video : {video_url}"
    else:
        return f"😔 Sorry, I couldn't find {name} on YouTube."


# send some metrics every 'time_period'
def send_metrics():
    global time_period
    global requests_counter
    global responses_time
    global process
    while True:
        cpu_usage = process.cpu_percent(interval=0)
        graphyte.send(
            "metrics_ChatWithMe-telegram-bot/CPU_usage_percent",
            cpu_usage,
        )
        # send the response time average metric for every request sent during 'time_period' seconds to the graphite server (carbon)
        # 2 responses metric
        graphyte.send(
            "metrics_ChatWithMe-telegram-bot/Responses_average",
            responses_time,
        )
        responses_time = 0
        # Send the received requests number during 'time_period' metric to the graphite server (carbon)
        # 1 requests metric
        graphyte.send(
            "metrics_ChatWithMe-telegram-bot/Requests_Number",
            requests_counter,
        )
        requests_counter = 0

        # Wait for 'time_period' seconds before sending the next metrics
        time.sleep(time_period)


# the responsible function for handling incoming requests
def handle_message(update, context):
    global requests_counter

    global responses_time
    # Increment the requests counter for every message received
    requests_counter += 1
    response_start_time = time.time()
    # Get the necessary informations
    global authentified_users
    global password

    user_id = update.effective_user.id
    UserName = update.effective_user.full_name
    # timestamp = time.time()
    message = update.message.text.lower()

    if user_id in authentified_users:
        if message.startswith("vid "):
            message = message.replace("vid ", "")
            answer = send_video(message, update.message.chat_id, Token)
            update.message.reply_text(answer)
            response_end_time = time.time()
            print(
                f"   processed response time : {response_end_time - response_start_time} seconds"
            )

            responses_time += response_end_time - response_start_time

            return

        # test if the user is requesting an image or a texted response
        if message.startswith("image "):
            message = message.replace("image ", "")
            prompt = f"Generate an image of {message}"
            response = openai.Image.create(
                model="image-alpha-001",
                prompt=prompt,
                n=1,
                size="512x512",
                response_format="url",
            )
            image_url = response["data"][0]["url"]
            # Download the image and save it to a local file
            image_name = f"{message}.jpg"
            image_path = os.path.join("uploads", image_name)
            with open(image_path, "wb") as f:
                f.write(requests.get(image_url).content)
            # Send the image to the user
            update.message.reply_photo(open(image_path, "rb"))
            response_end_time = time.time()
            print(
                f"   processed response time : {response_end_time - response_start_time} seconds"
            )
            responses_time += response_end_time - response_start_time

            return
        # check if the intended answer would be in arabic by checking the request question
        if message.startswith("ar "):
            message = message.replace("ar ", "")
            request = f"'{message}', give answer in Arabic language"
        else:
            request = f"'{message}'"
        if request in yourOwner:
            answer = "my owner is Mr. Imad belattar"
        elif request in aboutTheOwner:
            answer = owner
        elif message.startswith("sound "):
            message = message.replace("sound ", "")
            # Call the OpenAI Playground API to get a response
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=message,
                max_tokens=128,
                top_p=1,
                n=1,
                stop=None,
                temperature=0.7,
                frequency_penalty=0,
                presence_penalty=0,
            )
            answer = response.choices[0].text.strip()
            # convert the answer to an audio file using gTTS
            tts = gTTS(answer)
            audio_file = BytesIO()
            tts.write_to_fp(audio_file)
            audio_file.seek(0)
            # send the audio file to the user
            context.bot.send_voice(chat_id=user_id, voice=audio_file)
            response_end_time = time.time()
            print(
                f"   processed response time : {response_end_time - response_start_time} seconds"
            )

            responses_time += response_end_time - response_start_time

            return
        else:
            # Call the OpenAI Playground API to get a response
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=request,
                max_tokens=128,
                top_p=1,
                n=1,
                stop=None,
                temperature=0.7,
                frequency_penalty=0,
                presence_penalty=0,
            )
            answer = response.choices[0].text.strip()
        print(" ")
        print(f"  NR = {requests_counter} ,  User : {UserName}")
        print(f"   ******  question is : {request}")
        print(f"   ******  answer is : {answer}")
        update.message.reply_text(answer)
        response_end_time = time.time()
        print(
            f"   processed response time : {response_end_time - response_start_time} seconds"
        )
        responses_time += response_end_time - response_start_time
    else:
        if message == password:
            authentified_users[user_id] = "yes"
            
            answer = f"Great! Mr. {UserName} let's begin"
            #  print(f"{answer}")
            update.message.reply_text(answer)
            response_end_time = time.time()
            print(
                f"   processed response time : {response_end_time - response_start_time} seconds"
            )

            responses_time += response_end_time - response_start_time
        else:
            answer = f"Password is required to start the conversation :"
            # print(f"{answer}")
            update.message.reply_text(answer)
            response_end_time = time.time()
            print(
                f"   processed response time : {response_end_time - response_start_time} seconds"
            )

            responses_time += response_end_time - response_start_time


updater = telegram.ext.Updater(Token, use_context=True)
disp = updater.dispatcher
disp.add_handler(telegram.ext.CommandHandler("start", start))
disp.add_handler(telegram.ext.CommandHandler("help", help))
disp.add_handler(telegram.ext.MessageHandler(telegram.ext.Filters.text, handle_message))
# added
# Start the metric sending thread
metric_thread = threading.Thread(target=send_metrics)
metric_thread.daemon = True
metric_thread.start()
# /added

updater.start_polling()
updater.idle()

# buffered on 7:13 7/3/2023