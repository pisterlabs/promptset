import boto3
import openai
import os
import urllib
import sqlite3
import flickr_api
import random
from handling import QuoteAlreadyInDatabaseException
import google.generativeai as palm
from datetime import datetime
from dotenv import load_dotenv
from textwrap import fill
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# SETUP
load_dotenv()


ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")

s3 = boto3.client('s3',aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)


# PALM API
PALM_KEY = os.getenv("PALM_KEY")
palm.configure(api_key=PALM_KEY)

models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name

# FLICKR
FLICKR_KEY = os.getenv("FLICKR_KEY")
FLICKR_SKEY = os.getenv("FLICKR_SKEY")
flickr_api.set_keys(api_key=FLICKR_KEY, api_secret=FLICKR_SKEY)



def generate_quote(usr_prompt):
    try:
        quote = palm.generate_text(
            model=model,
            prompt=usr_prompt,
            temperature=1.0,
            max_output_tokens=300,
            stop_sequences="**",
        )
        add_to_database(quote.result)
        formatted_quote = fill(quote.result, 20)
        return formatted_quote
    except:
        if not check_for_duplicates(quote.result):
            raise QuoteAlreadyInDatabaseException



def get_image():
    file_name = str(datetime.now()) + ".png"
    folder_path = "/Users/hallowsyves/Documents/HourlyMotivation/Media/Images"
    file_path = os.path.join(folder_path, file_name)



    search_r = flickr_api.Photo.search(text='Nature Background', content_types=0, sorted=True)
    random_image = random.choice(search_r)
    flickr_api.Photo.save(self=random_image, filename=file_path, size_label='Medium 800')
    return file_path

    
def save_image(url):
    file_name = str(datetime.now()) + ".png"
    folder_path = "/Users/hallowsyves/Documents/HourlyMotivation/Media/Images"
    file_path = os.path.join(folder_path, file_name)
    urllib.request.urlretrieve(url, file_path)
    return file_path


def load_image(image, quote):
    # Load Background Image
    image_ = Image.open(image)
    image_.putalpha(127)

    # center text
    center_x = image_.width / 2
    center_y = image_.height / 2

    image_.filter(ImageFilter.GaussianBlur(5))
    image_load = ImageDraw.Draw(image_)

    # Draw Image
    font = load_font()
    image_load.text((center_x,center_y), quote, anchor='mm', font=font, fill=(255,255,255))

    # Show new Image with quote
    image_.save('Media/Images/temp.png')

    file_name = str(datetime.now()) + ".png"
    
    mimetype = 'image/png'
    
    s3.upload_file(
        Filename='Media/Images/temp.png',
        Bucket='hourlymotivationbgimg',
        Key=file_name,
        ExtraArgs={
            "ContentType": mimetype
        }
    )


    url = s3.generate_presigned_url('get_object',
                                    Params={
                                        'Bucket': 'hourlymotivationbgimg',
                                        'Key': file_name,
                                    },
                                    ExpiresIn=315360000)
    print(url)
    os.remove('Media/Images/temp.png')
    os.remove(image)
    return url


def load_font():
    times_new = ImageFont.truetype('/Users/hallowsyves/Documents/HourlyMotivation/Fonts/AUGUSTUS.TTF', 25)
    return times_new

def check_for_duplicates(quote):
    conn = sqlite3.connect('quotes.db')
    cursor = conn.cursor()
    query = 'SELECT * FROM {}'.format('quotes')
    cursor.execute(query)
    results = cursor.fetchall()
    for row in results:
        if quote in row:
            return False
    
    return True

def add_to_database(quote):
    connection = sqlite3.connect('quotes.db')
    cursor = connection.cursor()

    query = 'INSERT INTO quotes (quote) VALUES (?)'
    cursor.execute(query, (quote,))

    connection.commit()
    cursor.close()
    connection.close()


def print_database():
    connection = sqlite3.connect('quotes.db')
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM quotes')
    quotes = cursor.fetchall()

    for quote in quotes:
        print(quote)
    connection.close()

def generate_motivational_prompt():
    """1. Figure out how to create motivational prompt"""
    m_prompt = palm.generate_text(
        model=model,
        prompt="Pick one random historical figure, and one subject correlated to that figure",
        temperature=1.0,
        max_output_tokens=400,
        stop_sequences="**"
    )
    prompt = fill(m_prompt.result,20)
    return prompt

# Generate motivational quote, from rand person