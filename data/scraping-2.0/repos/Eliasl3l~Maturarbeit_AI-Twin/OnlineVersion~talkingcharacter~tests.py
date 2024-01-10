from django.test import TestCase
from . import secrets, utils
import openai
import os
import cv2
import urllib.request


class Secretkeytesting(TestCase):
    # original source: https://stackoverflow.com/questions/76522693/how-to-check-the-validity-of-the-openai-key-from-python#:~:text=To%20check%20if%20your%20OpenAI,Here%27s%20a%20simple%20code%20example
    def test_openaikey_isStillWorking(self):
        openai.api_key = secrets.OPENAI_KEY
        try:
            response = openai.Completion.create(
                engine="davinci",
                prompt="say 'Hello' in German",
                max_tokens=5
            )
        except Exception as E:
            self.fail(E)
        else:
            responseText = response.choices[0].text.strip()
            print(responseText)
            self = True
    
    def test_videoAnimation(self):
        try:
            responseVideo = utils.make_video("hello")
        except Exception as E:
            self.fail(E)
        else:
            #inspired by Chatgpt
            url = responseVideo
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            video_path = os.path.join(BASE_DIR, "talkingcharacter", "video.mp4")
            urllib.request.urlretrieve(url, video_path)

            try:
                cap = cv2.VideoCapture(video_path)
            except Exception as E:
                self.fail(E)


            while cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret:  # If no frame is captured, break the loop
                        break
                except Exception as E:
                    self.fail(E)
                cv2.imshow('Video', frame)
                
                # Exit on pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

            #delete the mp4 file if it exist or otherways give an error message
            if os.path.exists(video_path):
                os.remove(video_path)
                self = True
            else:
                self.fail("The video wasn't created in the proper folder")

            

# The video Test doesn't work yet because of an errror of the cv2 module
        
class UtilsTesting(TestCase):
    def GPTResponseTest(self):
        try:
            responseText = utils.get_chatgpt_response("say 'Hello' in German")
        except Exception as E:
            self.fail(E)
        else: 
            print(responseText)
            self = True
    
    
