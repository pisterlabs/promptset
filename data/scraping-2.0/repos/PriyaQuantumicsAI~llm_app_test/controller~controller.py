# Author: Priyadharshini Devarajan
# Date: September 21, 2023
# Description: Controller page for api jarvis_ai_process_text.


import openai
import requests
import json
# Creating a class for the controller
class AppController:

    def __init__(self, main_logger, config):
        self._logger = main_logger
        self._app_config = config

    def jarvis_gen_ai_process_text(self, json_data):
        res_dic = {}
        user_prompt = json_data.get('prompt')
        res_dic = {user_prompt : 'Success'}
        print(res_dic)
        return res_dic
        
    # write a function  to extract the project details from the json_data
    def extract_project_details(self, json_data):
        project_details = json_data.get('project_details')
        return project_details
            


    def call_chatgpt_api(prompt):
        url = 'https://api.openai.com/v1/engines/davinci-codex/completions'
        headers = {'Content-Type': 'application/json',
                'Authorization': 'Bearer YOUR_API_KEY'}
        data = {
            'prompt': prompt,
            'max_tokens': 100,
            'temperature': 0.7,
            'n': 1,
            'stop': None
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()
        else:
            return None