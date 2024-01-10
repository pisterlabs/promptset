#!/usr/bin/python3
import cgi
import os
import time
import boto3
import openai
import cv2
import numpy as np

upload_dir = "myupload/"
aws_access_key = 'AKIAQJYDURIUH7ASYGEB'
aws_secret_key = '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'
openai.api_key = 'sk-5FLnU0a4ORqB7dnnnnnnnjslk;sk;sk;lksl;sl;AoyEubs'

print("Content-Type: text/plain")
print()

def detect_labels(image_bytes):
    client = boto3.client('rekognition', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name='ap-south-1')
    response = client.detect_labels(Image={'Bytes': image_bytes})
    label_names = [label['Name'] for label in response['Labels']]
    return label_names

def generate_response(prompt):
    response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo-0301",
              messages=[{"role": "system", "content": 'You are a helpful assistant who is accompaning a blind person'},
                        {"role": "user", "content": prompt}
              ])
    return response.choices[0].message['content'].strip()

try:
    form = cgi.FieldStorage()
    image_file = form['image']

    if image_file.filename:
        timestamp = int(time.time())
        filename = f"image_{timestamp}.png"
        
        filepath = os.path.join(upload_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(image_file.file.read())
        print("Image uploaded successfully")

        with open(filepath, 'rb') as f:
            labels = detect_labels(f.read())
        prompt = "Image labels: " + ", ".join(labels) + "\n"
        user_prompt = "these are the labels extracted from an image using AWS Rekognition, use these labels to describe what's going around, note that you are helping a blind person understand the scene. Describe the scenario in a compassionate and keep the response short and simple, (don't exaggerate it, don't lie) try to be accurate\n"
        full_prompt = prompt + user_prompt

        response_text = generate_response(full_prompt)
        print(response_text)

    else:
        print("No image file received")
except Exception as e:
    print("An error occurred:", str(e))



try:
 
    with open(filepath, 'wb') as f:
        f.write(image_file.file.read())
    print("Image uploaded successfully")
except Exception as e:
    print("Error uploading image:", str(e))
