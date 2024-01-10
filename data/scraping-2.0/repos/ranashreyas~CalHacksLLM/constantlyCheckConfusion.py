import os
import subprocess
import pyautogui
import pytesseract
from PIL import ImageGrab
import time

import openai
import json
import requests
from bs4 import BeautifulSoup
import cv2

from hume import HumeBatchClient
from hume.models.config import FaceConfig
from pprint import pprint
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time

# import OCR


# Set up your OpenAI API credentials
openai.api_key = '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'

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
client = HumeBatchClient("bFDAvikKUskwAc1fKS22R7vuoimXeXAIncrBboEq3qp9LhhB")
config = FaceConfig()

# Function to make Google Chrome the active application
def activate_chrome():
    script = """
    tell application "Google Chrome"
        activate
    end tell
    """
    subprocess.call(['osascript', '-e', script])


# Main function
def main():
    activate_chrome()

    cap = cv2.VideoCapture(0)
    time.sleep(0.5)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    while True:
        
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


        cv2.imwrite('face.jpg', frame)

        job = client.submit_job(None, [config], files = ['face.jpg'])
        job.await_complete()
        status = job.get_status()
    #     print(f"Job status: {status}")

        details = job.get_details()
        run_time_ms = details.get_run_time_ms()
    #     print(f"Job ran for {run_time_ms} milliseconds")

        predictions = job.get_predictions()

        emotions = []

        try:
            # Extract emotions from the JSON
            for prediction in predictions[0]['results']['predictions']:
                for emotion in prediction['models']['face']['grouped_predictions'][0]['predictions'][0]['emotions']:
                    emotions.append(emotion)

            # Print the extracted emotions
            for emotion in emotions:
                if emotion['name'] == 'Confusion':
                    print(emotion['name'], emotion['score'])
                    if emotion['score'] > 0.45:
                        subprocess.run(['python', 'OCR2.py'])
                        break
                        
                        # # Get the active tab's URL and title
                        # tab_url = OCR.get_active_tab_url()
                        # tab_title = OCR.get_active_tab_title()
                        # print(f"Active Tab: {tab_title} ({tab_url})")
                        
                        # # Introduce a 5-second delay
                        # # time.sleep(3)
                        
                        # # Perform OCR on the active tab's content
                        # text = OCR.ocr_from_clipboard()
                        
                        # if text == "NOT CHROME":
                        #     continue
                        # if text:
                        #     text_input = text
                        #     # Provide your prompt here
                        #     prompt = "this is all the text on my screen, can you tell me what's wrong with my code and give me hints without giving away the full solution. Given that the site is leetcode, infer how you can help me: " + text_input

                        #     # Generate a response based on the prompt
                        #     response = OCR.generate_response(prompt)

                        #     # Print the generated response
                        #     print(response)
                        # else:
                        #     print("No text was extracted.")
        except: 
            print("no face detected, i think")

if __name__ == '__main__':
    main()
