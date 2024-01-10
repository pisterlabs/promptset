import datetime
import os
import tempfile
import uuid

import openai
import psycopg2
import speech_recognition as sr
import telebot
from aamarpay.aamarpay import aamarPay
from dotenv import load_dotenv
from gtts import gTTS
from pydub import AudioSegment
from telebot import types
from telebot.types import Message

load_dotenv()
BOT_TOKEN = os.getenv("BOT1_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
STORE_ID = os.getenv("STORE_ID")
SIGNATURE_KEY = os.getenv("SIGNATURE_KEY")
BACKEND_URL = os.getenv("BACKEND_URL")
BOT_ENV = os.getenv("BOT_ENV")
openai.api_key = OPENAI_API_KEY

bot = telebot.TeleBot(BOT_TOKEN)

# These functions use OpenAI API for grammar correction, paraphrasing and summarizing

user_states = {}  # To keep track of the previous converted message from voice to text
prev_states = {}  # To keep track if the previous message was voice or not
user_voice_durations = {}  # To keep track of the duration of the voice message


def grammar_correction(input_text):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Generate a grammar correction of the following sentence:\n\n{input_text}",
        temperature=0.5,
        max_tokens=200
    )
    return response.choices[0].text.strip()


def paraphrasing(input_text):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Generate a paraphrase of the following sentence:\n\n{input_text}",
        temperature=0.7,
        max_tokens=200
    )
    return response.choices[0].text.strip()


def summarizing(input_text):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Summarize the following paragraph:\n\n{input_text}",
        temperature=0.5,
        max_tokens=200
    )
    return response.choices[0].text.strip()


def count_words(text):
    return len(text.split())

# Function to calculate cost per message, considering 1000 tokens for 750 words


def calculate_cost(message, cost_per_thousand_tokens=0.02):
    num_words = count_words(message)
    num_tokens_estimate = (num_words / 750) * 1000
    cost_per_token = cost_per_thousand_tokens / 1000
    return num_tokens_estimate * cost_per_token


def initiate_payment(chat_id):
    transaction_id = str(uuid.uuid4())
    is_sandbox = True
    # Change the values based on the environment
    if BOT_ENV == 'production':
        is_sandbox = False
    pay = aamarPay(isSandbox=is_sandbox, transactionAmount=200,
                   transactionID=transaction_id)
    if BOT_ENV == 'production':
        pay.storeID = STORE_ID
        pay.signature = SIGNATURE_KEY
    pay.successUrl = f"{BACKEND_URL}/payment/success?chat_id=" + chat_id
    pay.failUrl = f"{BACKEND_URL}/payment/failed?chat_id=" + chat_id
    pay.cancelUrl = f"{BACKEND_URL}/payment/cancel?chat_id=" + chat_id
    payment_path = pay.payment()
    return payment_path


def count_user_phone_number(phone_number):
    conn = psycopg2.connect(f"{DATABASE_URL}")
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM users WHERE phone_number = %s", (phone_number,))
    count = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return count


def update_duration(time, id):
    conn = psycopg2.connect(f"{DATABASE_URL}")
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET duration_left = duration_left + %s WHERE user_id = %s", (time, id))
    conn.commit()
    cur.close()
    conn.close()


def update_free_usages(id):
    conn = psycopg2.connect(f"{DATABASE_URL}")
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET free_usages_left = free_usages_left - 1 WHERE user_id = %s", (id,))
    conn.commit()
    cur.close()
    conn.close()


def fetch_duration_left(id):
    conn = psycopg2.connect(f"{DATABASE_URL}")
    cur = conn.cursor()
    cur.execute(
        "SELECT duration_left FROM users WHERE user_id = %s", (id,))
    duration_left = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return duration_left


def fetch_paid_status(id):
    conn = psycopg2.connect(f"{DATABASE_URL}")
    cur = conn.cursor()
    cur.execute(
        "SELECT paid_status FROM users WHERE user_id = %s", (id,))
    paid_status = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return paid_status


def fetch_free_usages(id):
    conn = psycopg2.connect(f"{DATABASE_URL}")
    cur = conn.cursor()
    cur.execute(
        "SELECT free_usages_left FROM users WHERE user_id = %s", (id,))
    free_usages = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return free_usages


def save_user_data(user_id, username, phone_number):
    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(f"{DATABASE_URL}")

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    # Execute an SQL query to insert the user data
    query = "INSERT INTO users (user_id, username, phone_number) VALUES (%s, %s, %s)"
    values = (user_id, username, phone_number)
    cursor.execute(query, values)

    # Commit the transaction and close the connection
    conn.commit()
    cursor.close()
    conn.close()


def save_texts(text):
    conn = psycopg2.connect(f"{DATABASE_URL}")
    cur = conn.cursor()
    if text.startswith("Corrected"):
        updated_text = text.replace("Corrected: ", "")
        cur.execute(
            "UPDATE users SET corrected_texts = array_append(corrected_texts, %s)", (updated_text,))
    elif text.startswith("Paraphrased"):
        updated_text = text.replace("Paraphrased: ", "")
        cur.execute(
            "UPDATE users SET paraphrased_texts = array_append(paraphrased_texts, %s)", (updated_text,))
    elif text.startswith("Summarized"):
        updated_text = text.replace("Summarized: ", "")
        cur.execute(
            "UPDATE users SET summarized_texts = array_append(summarized_texts, %s)", (updated_text,))
    conn.commit()
    cur.close()
    conn.close()


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_animation(
        chat_id=message.chat.id,
        animation="https://edvive.s3.ap-southeast-1.amazonaws.com/dolphi2.gif",
    )
    bot.send_message(
        message.chat.id, "First things first, what's your name? Just type it in, and we'll be buddies in no time!")

    bot.register_next_step_handler(message, save_user_data_step)


def save_user_data_step(message):
    # Get the username entered by the user
    username = message.text

    # Create a custom keyboard
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
    phone_button = types.KeyboardButton(
        'Share Phone Number', request_contact=True)
    markup.add(phone_button)

    # Ask the user to share their phone number
    bot.send_message(
        message.chat.id,
        f"Nice to meet you, {username}! 🤗 Now, could you share your phone number with me? Don't worry, I won't spam you—I'm just excited to assist you!",
        reply_markup=markup
    )

    # Register the next step handler to handle the user's response
    bot.register_next_step_handler(message, save_phone_number_step, username)


def save_phone_number_step(message, username):
    phone_number = message.contact.phone_number
    user_id = int(str(message.chat.id)[-5:])
    payment_link = initiate_payment(str(message.chat.id))
    # duration_left = fetch_duration_left(user_id)
    # user_info = user_voice_durations[user_id]
    # duration = user_info["total_duration"] < 240*60

    # Add the keyword buttons
    markup = types.ReplyKeyboardMarkup(row_width=1)
    correction_button = types.KeyboardButton('Fix it')
    paraphrase_button = types.KeyboardButton('Improve it')
    # summary_button = types.KeyboardButton('Summary')
    markup.add(correction_button, paraphrase_button)

    # Existing users
    if count_user_phone_number(phone_number) >= 1:
        free_usages = fetch_free_usages(user_id)
        paid_status = fetch_paid_status(user_id)
        # Add the inline buttons
        subscribe_button = types.InlineKeyboardButton(
            "Subscribe Now", url=f"{payment_link}")
        free_usages_button = types.InlineKeyboardButton(
            f"{free_usages} free usages left", callback_data='free_usages')
        inline_markup = types.InlineKeyboardMarkup(
            [[subscribe_button, free_usages_button]])
        if paid_status == True:
            bot.send_message(
                message.chat.id,
                f"Welcome back, {username}! 🤗 I'm so glad you're here again!\n\nI have three nifty options lined up for you to leverage your English skills.\n\nWhich one sparks your interest?")
            bot.send_message(
                message.chat.id,
                "1️⃣ Correction: I'll polish your grammar and fix those sneaky mistakes.\n\n2️⃣ Paraphrase: Want to add some flair to your speech? I'll help you rephrase it in style!\n\nJust send me an audio, and we'll begin our language adventure! 🚀💬", reply_markup=markup)

        elif free_usages == 0 and paid_status == False:
            bot.send_message(message.chat.id,
                             f"Welcome back, {username}! 🤗 I'm so glad you're here again!\n\nYou have used up all your free usages. Please subscribe to continue using the service. 🙏🏻", reply_markup=inline_markup)
        else:
            bot.send_message(
                message.chat.id,
                f"Welcome back, {username}! 🤗 I'm so glad you're here again!\n\nI have three nifty options lined up for you to leverage your English skills.\n\nWhich one sparks your interest?",
                reply_markup=inline_markup)
            # Send another message with the reply keyboard
            bot.send_message(
                message.chat.id,
                "1️⃣ Correction: I'll polish your grammar and fix those sneaky mistakes.\n\n2️⃣ Paraphrase: Want to add some flair to your speech? I'll help you rephrase it in style!\n\nJust send me an audio, and we'll begin our language adventure! 🚀💬",
                reply_markup=markup
            )
    # New users
    else:
        save_user_data(user_id, username, phone_number)
        free_usages = fetch_free_usages(user_id)
        paid_status = fetch_paid_status(user_id)
        subscribe_button = types.InlineKeyboardButton(
            "Subscribe Now", url=f"{payment_link}")
        free_usages_button = types.InlineKeyboardButton(
            f"{free_usages} free usages left", callback_data='free_usages')
        inline_markup = types.InlineKeyboardMarkup(
            [[subscribe_button, free_usages_button]])
        bot.send_message(
            message.chat.id,
            f"Thanks for sharing,{username}! Ready to turn learning into adventure? 🚀\n\nCorrecting grammar glitches 📝 and spicing up speech with paraphrases 🔥 is my thing!\n\nBoost your English-speaking superpower, get 10 free mins daily, or go unlimited for 3000 BDT/month!\n\nLet's conquer new horizons together! 🌐 Share your speech as audio and Let's get started!",
            reply_markup=inline_markup
        )

        # Send another message with the reply keyboard
        bot.send_message(
            message.chat.id,
            "1️⃣ Correction: I'll polish your grammar and fix those sneaky mistakes.\n\n2️⃣ Paraphrase: Want to add some flair to your speech? I'll help you rephrase it in style!\n\nJust send me an audio, and we'll begin our language adventure! 🚀💬",
            reply_markup=markup
        )

    # Reset the user's state
    user_states[message.chat.id] = None


@bot.message_handler(func=lambda msg: True, content_types=['text', 'audio', 'photo', 'video', 'document', 'sticker', 'video_note', 'poll', 'location', 'contact'])
def handle_text(message):
    text = user_states.get(message.chat.id)
    prev = prev_states.get(message.chat.id)
    user_id = int(str(message.chat.id)[-5:])
    payment_link = initiate_payment(str(message.chat.id))
    free_usages = fetch_free_usages(user_id)
    paid_status = fetch_paid_status(user_id)
    # print(message.chat.id)

    if free_usages == 0 and paid_status == False:
        subscribe_button = types.InlineKeyboardButton(
            "Subscribe Now", url=f"{payment_link}")
        markup = types.InlineKeyboardMarkup([[subscribe_button]])
        bot.reply_to(
            message, "Free usage complete! 🎉 Unlock unlimited progress with our subscription - 3000 BDT/month for 240 mins daily. Supercharge your English-speaking skills now! 💪🚀", reply_markup=markup)
        return

    if message.content_type == 'text' and prev == 'voice':
        input_text = message.text.lower()
        update_free_usages(user_id)
        free_usages = fetch_free_usages(user_id)

        if input_text == 'fix it':
            corrected_text = grammar_correction(text)
            formatted_corrected_text = f"Fixed: {corrected_text}"
            tts = gTTS(text=formatted_corrected_text, lang='en')
            tts.save('corrected_text.mp3')
            audio = open('corrected_text.mp3', 'rb')
            # Rendering buttons based on conditions
            markup = types.ReplyKeyboardMarkup(row_width=1)
            if not paid_status:
                subscribe_button = types.InlineKeyboardButton(
                    "Subscribe Now", url=f"{payment_link}")
                free_usages_button = types.InlineKeyboardButton(
                    f"{free_usages} free usages left", callback_data='free_usages')
                markup = types.InlineKeyboardMarkup(
                    [[subscribe_button, free_usages_button]])
            bot.send_voice(chat_id=message.chat.id, voice=audio)
            bot.reply_to(message, formatted_corrected_text,
                         reply_markup=markup)
            audio.close()  # Close the file before deleting
            os.remove('corrected_text.mp3')
            save_texts(formatted_corrected_text)

        elif input_text == 'improve it':
            paraphrased_text = paraphrasing(text)
            formatted_paraphrased_text = f"Improved: {paraphrased_text}"
            tts = gTTS(text=formatted_paraphrased_text, lang='en')
            tts.save('paraphrased_text.mp3')
            audio = open('paraphrased_text.mp3', 'rb')
            # Rendering buttons based on conditions
            markup = types.ReplyKeyboardMarkup(row_width=1)
            if not paid_status:
                subscribe_button = types.InlineKeyboardButton(
                    "Subscribe Now", url=f"{payment_link}")
                free_usages_button = types.InlineKeyboardButton(
                    f"{free_usages} free usages left", callback_data='free_usages')
                markup = types.InlineKeyboardMarkup(
                    [[subscribe_button, free_usages_button]])
            bot.send_voice(chat_id=message.chat.id, voice=audio)
            bot.reply_to(message, formatted_paraphrased_text,
                         reply_markup=markup)
            audio.close()  # Close the file before deleting
            os.remove('paraphrased_text.mp3')
            save_texts(formatted_paraphrased_text)

        # elif input_text == 'summary':
        #     summarized_text = summarizing(text)
        #     formatted_summarized_text = f"Summarized: {summarized_text}"
        #     tts = gTTS(text=formatted_summarized_text, lang='en')
        #     tts.save('summarized_text.mp3')
        #     audio = open('summarized_text.mp3', 'rb')
        #     # Rendering buttons based on conditions
        #     markup = types.ReplyKeyboardMarkup(row_width=1)
        #     if not paid_status:
        #         subscribe_button = types.InlineKeyboardButton(
        #             "Subscribe Now", url=f"{payment_link}")
        #         free_usages_button = types.InlineKeyboardButton(
        #             f"{free_usages} free usages left", callback_data='free_usages')
        #         markup = types.InlineKeyboardMarkup(
        #             [[subscribe_button, free_usages_button]])
        #     bot.send_voice(chat_id=message.chat.id, voice=audio)
        #     bot.reply_to(message, formatted_summarized_text,
        #                  reply_markup=markup)
        #     audio.close()  # Close the file before deleting
        #     os.remove('summarized_text.mp3')
        #     save_texts(formatted_summarized_text)

        if free_usages == 1 and paid_status == False:
            bot.send_message(message.chat.id,
                             "You have 1 free usage left. Please subscribe to continue using the service. 🙏🏻")

        user_states[message.chat.id] = None
        prev_states[message.chat.id] = None
    else:
        bot.reply_to(
            message, "Audio please! 🎙️💬 It helps me enhance your spoken English. Let's improve together! 🌟🗣️")


@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    user_id = int(str(message.chat.id)[-5:])
    free_usages = fetch_free_usages(user_id)
    paid_status = fetch_paid_status(user_id)

    voice_duration = message.voice.duration
    print(voice_duration)

    # If user is not in the dictionary, add them
    if user_id not in user_voice_durations and paid_status == True:
        user_voice_durations[user_id] = {
            "total_duration": 0,
            "start_date": datetime.datetime.today()
        }
        user_info = user_voice_durations[user_id]

    # If more than an hour has passed since the last message, reset the total duration
    if paid_status == True and datetime.date.today() != user_info["start_date"]:
        user_info["total_duration"] = 0
        user_info["start_date"] = datetime.datetime.today()

    # If the total duration with the new message exceeds 240 minutes, send a warning
    if paid_status == True and user_info["total_duration"] + voice_duration > 240*60:
        bot.reply_to(message, "You have reached the limit for the day.")
        return
    elif paid_status == True:
        user_info["total_duration"] += voice_duration
        # print(user_info["total_duration"])

    # Add the inline buttons
    payment_link = initiate_payment(str(message.chat.id))
    subscribe_button = types.InlineKeyboardButton(
        "Subscribe Now", url=f"{payment_link}")
    free_usages_button = types.InlineKeyboardButton(
        f"{free_usages} free usages left", callback_data='free_usages')
    inline_markup = types.InlineKeyboardMarkup(
        [[subscribe_button, free_usages_button]])

    if free_usages == 0 and paid_status == False:
        payment_link = initiate_payment(str(message.chat.id))
        subscribe_button = types.InlineKeyboardButton(
            "Subscribe Now", url=f"{payment_link}")
        markup = types.InlineKeyboardMarkup([[subscribe_button]])
        bot.reply_to(
            message, "You have used up all your free usages. Please subscribe to continue using the service. 🙏🏻", reply_markup=markup)
        return
    # Download the voice message file
    voice_file = bot.get_file(message.voice.file_id)

    # Create a temporary file to save the voice message
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file_path = temp_file.name

    # Download the voice message file to the temporary location
    downloaded_file = bot.download_file(voice_file.file_path)

    # Save the downloaded file to the temporary location
    with open(file_path, 'wb') as f:
        f.write(downloaded_file)

    # Convert the audio file to the WAV format using pydub
    audio = AudioSegment.from_file(file_path, format="ogg")
    # Calculate the duration of the audio file
    # duration = int(len(audio) / 1000)  # Duration in seconds
    # update_duration(duration, user_id)
    # # See how much duration is left
    # duration_left = fetch_duration_left(user_id)
    # if duration_left <= 0:
    #     bot.reply_to(message.chat.id,
    #                  "You have used up all your full duration. Please wait to continue using the service. 🙏🏻")

    wav_file_path = file_path + ".wav"
    audio.export(wav_file_path, format="wav")

    # Convert the voice file to text using speech_recognition
    r = sr.Recognizer()
    with sr.AudioFile(wav_file_path) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)

    # Pass the text to the existing message handler function
    message.text = text
    user_states[message.chat.id] = text
    prev_states[message.chat.id] = 'voice'

    markup = types.ReplyKeyboardMarkup(row_width=1)
    correction_button = types.KeyboardButton('Fix it')
    paraphrase_button = types.KeyboardButton('Improve it')
    # summary_button = types.KeyboardButton('Summary')
    markup.add(correction_button, paraphrase_button)

    bot.send_message(message.chat.id, "Awesome audio you've got there! 🎧 Now, let's take it to the next level! How would you like me to work my magic on this audio to boost your English speaking skills? 🌟")

    bot.send_message(
        message.chat.id,
        "Choose your English upgrade: 'Fix It' for grammar magic 🧙‍♂️ or 'Improve It' for captivating paraphrases! ✨ Both boost speaking skills! Ready to work wonders? 🚀 Pick now and let's soar! 💫",
        reply_markup=markup
    )


bot.infinity_polling(timeout=10, long_polling_timeout=5)
