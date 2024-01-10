import csv
import requests
import os
from dotenv import load_dotenv
import openai
import re
import json
from time import sleep
import shutil
import ast

def encode_image(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def send_to_openai(messages):
    load_dotenv()
    """Send messages to OpenAI API and get a response."""
    headers = {
        'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": messages,
        "max_tokens": 2000
    }

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=payload
    )
    return response.json()

csv_file = []
dataset = os.listdir('dataset')
dataset.sort()
print(len(dataset))

CSV_FILE = 'twitter_data.csv'
header = ['Post_ID', 'User_Handle', 'Post_Content', 'Post_Timestamp', 'Post_Views', 'Comment_ID', 'Comment_Content', 'Comment_Timestamp', 'Comment_Views']

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header

for filename in dataset:
    messages = [{"role": "system", "content": "You are an image transcriber and interpreter."},{"role": "user", "content": "From now on, you are my personal tweet transcriber. What I'll do is sending you an image of a tweet. Transcribe these images into this format. If inside these post contains an image, ignore the image. If a user handle is not visible, send the rest of the data without the {'User_Handle'}. This is the format. {'User_Handle':user handle, 'Post_Content': message in the post, 'Post_Timestamp': time for example: 3:00 29/11/23, 'Post_Views': number of views} For timestamps do not add leading zero and same for dates. And for views, only keep the number and suffix but don't need to add the word Views. Lastly, if the comment contains emoji or special characters (i.e. enters ('\n') remove them and retrieve the message as one big paragraph."}, {"role": "assistant", "content": "Understood! You can send me images of tweets, and I'll transcribe them into the specified format. If the tweet includes an image, I'll ignore it. If a user handle is not visible, I'll exclude it from the data. Additionally, I'll ensure that timestamps don't have leading zeros, only keep the number and suffix for views (without the word views), and remove any emojis or special characters like line breaks, presenting the message as a single paragraph. Please go ahead and send the image of the tweet you'd like transcribed."}]
    # Use regular expression to find the pattern of interest in the filenames
    match = re.match(r'POST_(\d{3})_(\d+).(jpg|JPG|png|PNG)$', filename)
    if match:
        # Extract the post number and the comment number
        post_number = int(match.group(1))
        comment_number = int(match.group(2))
        # Append a tuple with the post number and comment number to the list
        post_tuple = (post_number, comment_number)
        print(filename, post_tuple)
    else:
        print(filename, 'did not match')
        continue

    img = encode_image(f'dataset/{filename}')
    messages.append({"role": "user", "content": [{"type": "img_url", "image_url": { "url": f"data:image/png;base64,{img}" }}]})
    response = send_to_openai(messages)
    print(response)
    # Error handling for the JSON response
    try:
        done_folder = os.path.join('dataset', 'done')
        if not os.path.exists(done_folder):
            os.makedirs(done_folder)

        shutil.move(os.path.join('dataset', filename), os.path.join(done_folder, filename))
        
        data = response['choices'][0]['message']['content']
        data = data.replace("'", '"')
        post_json_dict = json.loads(data)
        print(post_json_dict, type(post_json_dict))
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        continue
    
    try:
        if not post_tuple[1]: # if comment number is 0 (OG post)
            User_Handle, Post_Content, Post_Timestamp, Post_Views = post_json_dict['User_Handle'], post_json_dict['Post_Content'], post_json_dict['Post_Timestamp'], post_json_dict['Post_Views']
        else: # if comment number is not 0 (comment)
            csv_entries = (post_tuple[0], User_Handle, Post_Content, Post_Timestamp, Post_Views, post_tuple[1], post_json_dict['Post_Content'], post_json_dict['Post_Timestamp'], post_json_dict['Post_Views'])
            print(csv_entries)
            with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                #Write csv_entries to the csv row
                writer.writerow(csv_entries)
        sleep(30)
    except NameError as e:
        print(f"Name error: {e}")
        continue
    except KeyError as e:
        print(f"Key error: {e}")
        continue