import cv2
import openai

import openai
import os
from dotenv import load_dotenv


class Commentator:

    text_location = (25,550)
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_thickness = 3
    text_color = (255,255, 255)
    line_type = 3
    commentary_frames = -1

    current_commentary = ''
    # in case of API errors
    default = {'point': 'Fantastic shot by the player! That was a great display of agility, quick reflexes, and excellent technique. We can expect an exciting match today!',
               'serve': 'And we are off with a strong serve from the player!',
               'replay': 'Now let\'s check out a replay of that last rally'}

    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.events = {'point': 'A player has just scored a point. Keep it generic', 
                       'serve': 'A player has just served the ball',
                       'replay': 'A replay of the last point is being shown.  Don\'t talk about the specifics of the rally'}

    def set_commentary(self, event):
        
        # start displaying commentary
        self.commentary_frames = 0

        # The system parameter sets up the chat agent and gives it context for the conversation
        # user parameter is is a prompt from the user to the chat agent
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "You are a helpful commentator for a tennis match."},
                        {"role": "user", "content": f'{self.events[event]}'},
                    ]
                )
            self.current_commentary = response['choices'][0]['message']['content']
            print(self.current_commentary)
        except Exception as e:
            print(e)
            self.current_commentary = self.default[event]
        
        
    # TODO format this text nicely on screen
    def display_commentary(self, image):

        if self.commentary_frames <= 100 and self.commentary_frames >=0:
            cv2.putText(image, self.current_commentary, 
            self.text_location, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            self.font_scale,
            self.text_color,
            self.text_thickness,
            self.line_type)
            self.commentary_frames += 1
        else:
            self.commentary_frames = -1
        
