import telebot
import openai
from datetime import datetime, time
import os

# Set up the OpenAI API client
openai.api_key = "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"
TELEGRAM_API_KEY = ('6260954011:AAFiV-8H94nFg1PgTUEU1VtPELPjqc0bn3M')
TELEGRAM_CHANNEL_NAME = ('@tintaichinhbeta')
SCRAPING_INTERVAL_SECONDS = 5

# Create a Telegram bot instance
bot = telebot.TeleBot(TELEGRAM_API_KEY)

# Define a function to summarize news articles
def summarize_news(data):
    print('start summarizing data')
    today = datetime.today().strftime('%d/%m/%Y')
    # Use the OpenAI GPT API to summarize the article
    summary = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {'role': 'user',
             'content':
               "Today is " + today + "."
                + "Write a well-designed summary with headings and subheadings based on what you can understand from this in Vietnamese:\n"
                + data},
        ],
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.5
    )
    print('finish summarizing data\n')
    return summary['choices'][0]['message']['content']

import requests
# Define a function to send the summary to Telegram
def send_summary_to_telegram(summary, title, image_links, url):
    print('start sending summary to telegram')
    # Get the latest article from the database
    message = ''
    message += f'<b>{title}</b>\n - <b>A.I tóm tắt</b>:{summary}\n - Xem chi tiết: <a href="Link báo">{url}</a>\n'
    bot.send_message(chat_id=TELEGRAM_CHANNEL_NAME, text=message, parse_mode='HTML')

    for image_link in image_links[0]:
        response = requests.get(image_link
                                
                                )
        if response.status_code == 200:
            with open('image.jpg', 'wb') as image_file:
                image_file.write(response.content)
            with open('image.jpg', 'rb') as image_file:
                bot.send_photo(chat_id=TELEGRAM_CHANNEL_NAME, photo=image_file)
            os.remove('image.jpg')
    print('finish sending summary to telegram\n')


# Define a function to get new items from the database
from pymongo import MongoClient
uri = 'mongodb+srv://truongbodoi821:iCNPwul82lZuC4DM@cluster0.9fcp1gg.mongodb.net/?retryWrites=true&w=majority'
client = MongoClient(uri)
db = client['News']
collection = db['NewArticles']

def get_new_item():
    print('start getting new items')
    # Retrieve new items from the collection
    new_item = collection.find_one_and_update(
        {'summarized': False},
        {'$set': {'summarized': True}}
    )
    # Return the new items
    print('finish getting new items\n')
    return new_item

while True:
    # Get new items from the database
    print('going to loop main function')
    new_item = get_new_item()

    if new_item is None:
        print('no new items')
        continue
    summary = summarize_news(new_item['content'])
    tiltle = new_item['title']
    image_links = new_item['imgage_link']
    url = new_item['url']
    # Send the summary to Telegram
    send_summary_to_telegram(summary, tiltle, image_links, url)
    
