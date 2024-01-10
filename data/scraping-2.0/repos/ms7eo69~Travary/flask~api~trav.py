import openai
import os
from flask_restful import reqparse, Resource
from flask import Flask, request, jsonify
import time

# Set OpenAI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')

messages = [
            {
                "role": "system",
                "content": ("You are a professional travel assistant specializing in Korea. "
                            "Your name is Trav, and you explain about traveling in Korea. "
                            "You must treat all users with courtesy. "
                            "If there's a question for which you do not have a clear answer prepared, "
                            "please respond with \"Please inquire with the administrator.\" "
                            "You are a Korean-speaking bot, and the users are Koreans. "
                            "You must use Korean. "
                            "If a user asks you a question that is not related to traveling in Korea, "
                            "as Trav, you should clarify that you are a professional travel assistant "
                            "and encourage the user to ask questions related to traveling. "
                            "Always remember that you are Trav.")
            }
        ]

class TRAV(Resource):
    def chatbot(self, content, model="gpt-3.5-turbo", temperature=0):
        global messages  # declare that we are using the global variable

        try:
            messages.append({'role': 'user', 'content': content})
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            answer = response.choices[0].message.content
            messages.append({'role': 'assistant', 'content': answer})
            return {'status': 'SUCCESS', 'messages': messages}

        except openai.error.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            return {'status': 'FAIL', 'messages': e}

    def reset(self):
        global messages
        messages = [
            {
                "role": "system",
                "content": ("You are a professional travel assistant specializing in Korea. "
                            "Your name is Trav, and you explain about traveling in Korea. "
                            "You must treat all users with courtesy. "
                            "If there's a question for which you do not have a clear answer prepared, "
                            "please respond with \"Please inquire with the administrator.\" "
                            "You are a Korean-speaking bot, and the users are Koreans. "
                            "You must use Korean. "
                            "If a user asks you a question that is not related to traveling in Korea, "
                            "as Trav, you should clarify that you are a professional travel assistant "
                            "and encourage the user to ask questions related to traveling. "
                            "Always remember that you are Trav.")
            }
        ]
        return {'status': 'SUCCESS', 'message': 'Chat reset successfully.'}

    def post(self):
        data = request.get_json()
        if 'reset' in data and data['reset']:
            return self.reset()

        content = data['content']
        response = self.chatbot(content)
        return response