from tasks.abstractTask import AbstractTask
from openai import OpenAI
import logging
import os

class GnomeTask(AbstractTask):
    
    def __init__(self, task_signature: str, send_to_aidevs: bool, mock: bool):
        super().__init__(task_signature, send_to_aidevs, mock)
        self.OPEN_AI_CLINET = OpenAI()
        self.logger = logging.getLogger(__name__)
        
    def solve_task(self):
        return super().solve_task()
    
    def process_task_details(self):
        gnome_url = self.assignment_body["url"]
        # let gpt-4V assess if the picture from url is gnome or not. If it is - return gnome hat color in polish - otherwise return None
        user_meassage = "Check if there is gnome in the picture. If so - return gnome hat color in polish. Otherwise return word ERROR. Return NO or color in polish and nothing else"
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_meassage
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": gnome_url 
                        }
                    }
                ]
            }
        ]
        params = {
            "model": "gpt-4-vision-preview",
            "messages": prompt_messages,
            # "response_format": {"type": "json_object"},  # Added response format
        }
        response = self.OPEN_AI_CLINET.chat.completions.create(**params)       
        
        return response.choices[0].message.content