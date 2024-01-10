import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import http.client
import mimetypes
from codecs import encode
import openai
openai.api_key = "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"

# Connect to database
conn = sqlite3.connect('names.db')
cursor = conn.cursor()

names = []

# Create function to add a name


def add_name(english_name, arabic_name):
    try:
        cursor.execute('''INSERT INTO translated_names (english_name, arabic_name)
                        VALUES (?, ?)''', (english_name, arabic_name))
        conn.commit()
        print(english_name+" added successfully.")
    except:
        print("Error adding name.")


def read_names():
    try:
        cursor.execute('''SELECT * FROM translated_names 
                        WHERE translated_names.arabic_name like '' ''')
        rows = cursor.fetchall()
        return rows
    except:
        print("Error reading names.")


def update_name(english_name, arabic_name):
    try:
        cursor.execute('''UPDATE translated_names SET arabic_name = ? WHERE english_name = ?''',
                       (arabic_name, english_name))
        conn.commit()
        print("Name updated successfully.")
    except:
        print("Error updating name.")


def get_english_names():
    for i in range(1, 87):
        print("=================================\n             step:{}             \n=================================\n".format(i))
        url = 'https://www.behindthename.com/names/'+str(i)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for name in soup.select('.listname'):
            add_name(name.text, '')


def scrape_arabic_names(english_name):
    conn = http.client.HTTPSConnection("learnarabic.me")
    dataList = []
    boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=s;'))

    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))

    dataList.append(encode(english_name))
    dataList.append(encode('--'+boundary+'--'))
    dataList.append(encode(''))
    body = b'\r\n'.join(dataList)
    payload = body
    headers = {
        'Content-type': 'multipart/form-data; boundary={}'.format(boundary)
    }
    conn.request("POST", "/c48cfa005f53e9683ad438b6265385bb.php",
                 payload, headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")


def get_arabic_names():
    print("=================================\n          Read English Names         \n=================================\n")
    all_names = read_names()
    for name in all_names:
      try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a chatbot"},
                {"role": "user",
                 "content": "type this name in arabic (whitout any Description): {}".format(name)},
            ]
        )

        result = ''
        for choice in response.choices:
            result += choice.message.content

        update_name(name[0], result)
        print(name[0]+" ==> "+result)
        time.sleep(5)
      except Exception as e:
          print("beeb")


get_arabic_names()
