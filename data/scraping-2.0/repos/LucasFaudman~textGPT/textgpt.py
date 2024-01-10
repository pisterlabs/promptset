
import os
from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client

from messagedb import MessageDB, Message
from datetime import datetime

from imageprocessor import get_image_bytes_if_valid

import openai
from openai.error import ServiceUnavailableError, InvalidRequestError

TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

""""""


DEFAULT_SETTINGS = {'model_id':18, #gpt3.5-turbo 
                    'system_prompt_id': 1,
                    'stop_sequence': None,
                    'max_tokens': None,
                    'temperature': 1.0,
                    'top_p': 1.0,
                    'frequency_penalty': 0.0,
                    'presence_penalty': 0.0
                    }

HELP_STR = """TextGPT Commands:
#help
- prints this help message

#get [<key>|'settings'] 
- get your current setting for <key> or all settings

#set <key> <value>
- set your current setting for <key> to <value>

Keys for #get and #set:
- model: the OpenAI model to use 
- system_prompt: the prompt to use to insruct the model of what to do
- stop_sequence: the sequence of tokens to use to stop the model from generating more tokens
- max_tokens: the maximum number of tokens to generate
- temperature: the sample temperature to use when generating tokens (higher = more random)
- top_p: the nucleus sampling probability to use when generating tokens (higher = more random)
- frequency_penalty: the frequency penalty to use when generating tokens (higher = more conservative)
- presence_penalty: the presence penalty to use when generating tokens (higher = more conservative)

#reset ['all'|'messages'|'settings']
- all: resets everything
- messages: resets all messages
- settings: resets all settings

#models
- prints all available models

#limits
- prints your current usage limits

#image <'create'|'edit'|'variation'> <prompt>
- create: creates an image from the prompt
- edit: edits an image from the prompt
- variation: creates a new variation of an image 
"""


class textGPT(object):
    
    def __init__(self, db_name='test.db', number=TWILIO_PHONE_NUMBER) -> None:
        self.db_name = db_name
        self.client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        self.number = number
        self.mdb = MessageDB(self.db_name)
        self.mdb.add_phone_number(TWILIO_PHONE_NUMBER)
        self.update_models()



    def handle_incoming_message(self, message_values):
        incoming_message_sid = message_values.get('MessageSid', None)
        from_phone_number = message_values.get('From', None)
        to_phone_number = message_values.get('To', None)
        incoming_message_body = message_values.get('Body', None)
        media_url =  media_url = message_values.get('MediaUrl0', None)
    

        self.mdb = MessageDB(self.db_name)
        incoming_message_obj = self.mdb.add_message(incoming_message_sid, from_phone_number, to_phone_number, incoming_message_body)
        settings = self.mdb.get_settings_for_phone_number(from_phone_number)
        
        # If settings is None, add default settings
        if settings is None:
            self.mdb.add_settings(from_phone_number, **DEFAULT_SETTINGS)
            settings = self.mdb.get_settings_for_phone_number(from_phone_number)
        
        # Handle Commands ie #get settings #set settings #get system prompt #set system prompt
        if incoming_message_body.startswith('#'):
            outgoing_message_body = self.handle_command(incoming_message_obj, settings, media_url)
        else:
            outgoing_message_body = self.get_message_response(incoming_message_obj, settings)
        
        
        outgoing_message_objs = self.send_message(to_phone_number=from_phone_number, body=outgoing_message_body)
        
        return incoming_message_obj and outgoing_message_objs


    def send_message(self, to_phone_number, body, chunk_size=1600):
        if len(body) > chunk_size:
            chunks = [body[i:i+chunk_size] for i in range(0, len(body), chunk_size)]
        else:
            chunks = [body,] 

        self.mdb = MessageDB(self.db_name)
        outgoing_message_objs = []
        for chunk in chunks:
            if chunk.startswith('http'):
                media_url = chunk
                chunk = None
            else:
                media_url = None

            message = self.client.messages.create(
                from_=self.number,
                to=to_phone_number,
                body = chunk,
                media_url = media_url)

            message_sid = message.sid
            outgoing_message_obj = self.mdb.add_message(message_sid, self.number, to_phone_number, body)
            outgoing_message_objs.append(outgoing_message_obj)

        return outgoing_message_objs
    

        
    def get_message_response(self, incoming_message_obj, settings={}):
        user_phone_number = incoming_message_obj.from_phone_number
        text_messages = self.mdb.get_messages_for_phone_number(user_phone_number)

        openai_messages = [{'role': 'system', 'content': settings.pop('system_prompt', None)},]
        for message_obj in text_messages:
            if message_obj.body.startswith('#'):
                continue
            elif message_obj.from_phone_number == user_phone_number:
                role = 'user'
            elif message_obj.from_phone_number == self.number:
                role = 'assistant' 
            else:
                print('Error: message_obj.from_phone_number not in [user_phone_number, self.number]')
            
            openai_messages.append({'role': role, 'content': message_obj.body})

        settings['stop'] = settings.pop('stop_sequence', None)
        return self.openai_get_chat(openai_messages, **settings)




    def handle_command(self, incoming_message_obj, settings={}, media_url=None):
        verbs = ['help', 'get', 'set', 'reset', 'models', 'limits', 'image']
        settings_params = ['model', 'system_prompt', 'stop_sequence',
                  'max_tokens', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty']
        

        # Handle Commands ie #get settings #set settings #get system prompt #set system prompt
        user_phone_number = incoming_message_obj.from_phone_number
        
        command =  incoming_message_obj.body[1:].lower()
        verb, *args = command.split(maxsplit=2)

        #Assume command is invalid
        reply =  f'Error: {command} is not a valid command.'
        
        if verb == 'help' and len(args) == 0:
            reply = HELP_STR

        elif verb == 'get' and len(args) == 1:
            param = args[0]
            if param == 'settings':
                settings_str = '\n'.join(f'{k}:\n{v}' for k,v in settings.items())
                reply = f'Your settings are:\n{settings_str}'
            elif param in settings_params:
                reply = f'Your current {param} is:\n{settings[param]}'

        elif verb == 'set' and len(args) == 2:
            param, value = args
            if param in settings_params:
                self.mdb.update_settings_for_phone_number(user_phone_number, **{param: value})
                reply = f'Your {param} has been set to:\n{value}'
        
        elif verb == 'reset' and len(args) < 2:
            # If no args, reset all
            args = args or ['all',]
            param = args[0]            
            if param == 'settings':
                self.mdb.update_settings_for_phone_number(user_phone_number, **DEFAULT_SETTINGS)
                reply = f'Your settings have been reset to defaults'  
            elif param == 'messages':
                self.mdb.delete_messages_for_phone_number(user_phone_number)
                reply = f'Your conversation has been reset.'
            elif param == 'all':
                self.mdb.update_settings_for_phone_number(user_phone_number, **DEFAULT_SETTINGS)
                self.mdb.delete_messages_for_phone_number(user_phone_number)
                reply = f'Your settings and conversation have been reset.'
        
        elif verb == 'models' and len(args) == 0:
            reply = 'Available Models:\n' + '\n'.join(self.all_models)

        elif verb == 'limits' and len(args) == 0:
            reply = self.get_user_limits(user_phone_number)
        
        elif verb == 'image':
            param = args[0]
            prompt = args[1] if len(args) == 2 else ''
            
            if param == 'create':
                reply = self.openai_get_image(prompt)
            elif param == 'edit':
                reply = self.openai_get_image_edit(prompt, media_url=media_url)
            elif param == 'variation':
                reply = self.openai_get_image_variation(prompt, media_url=media_url)

        return reply
        

    def update_models(self):
        self.all_models = sorted([model['id'] for model in openai.Model.list()['data']])
        for model in self.all_models:
            self.mdb.add_model(model)
        return self.all_models


    def try_openai(self, getter_fn, parser_fn, **kwargs):
        try:
            response = getter_fn(**kwargs)
            return parser_fn(response)
        except Exception as e:
            if isinstance(e, ServiceUnavailableError):
                print(e)
                return f'Error: {e}'  
                #return 'ServiceUnavailableError'
            elif isinstance(e, InvalidRequestError):
                print(e)
                return f'Error: {e}'
                #return 'InvalidRequestError'
            else:
                print(e)
                return str(e)    

    def openai_get_chat(self, messages=[], n=1, **kwargs):
        
        result = self.try_openai(
            getter_fn=openai.ChatCompletion.create, 
            parser_fn=lambda response: response.choices[0].message.content.strip(),
            messages=messages, 
            n=n, **kwargs)

        return result

    def parse_size_from_prompt(self, prompt, default_size='256x256'):
        for size in ['256x256', '512x512', '1024x1024']:
            if size in prompt:
                return size, prompt.replace(size, '')
        
        return default_size, prompt


    def openai_get_image(self, prompt, n=1, **kwargs):
        size, prompt = self.parse_size_from_prompt(prompt)
        result = self.try_openai(
            getter_fn=openai.Image.create, 
            parser_fn=lambda response: response.data[0].url,
            size=size,
            prompt=prompt, 
            n=n, **kwargs)

        return result
    
    def openai_get_image_edit(self, prompt, media_url, n=1, **kwargs):
        size, prompt = self.parse_size_from_prompt(prompt)
        
        image_bytes, mask_bytes = get_image_bytes_if_valid(media_url)
        if not isinstance(image_bytes, bytes):
            error_message = image_bytes
            return error_message 

        result = self.try_openai(
            getter_fn=openai.Image.create_edit, 
            parser_fn=lambda response: response.data[0].url,
            size=size,
            prompt=prompt,
            image=image_bytes,
            mask=mask_bytes, 
            n=n, **kwargs)

        return result
    
    def openai_get_image_variation(self, prompt, media_url, n=1, **kwargs):
        size, prompt = self.parse_size_from_prompt(prompt)
        #n = int(prompt.strip()) if prompt else 1
        image_bytes, mask_bytes = get_image_bytes_if_valid(media_url)
        if not isinstance(image_bytes, bytes):
            error_message = image_bytes
            return error_message 

        result = self.try_openai(
            getter_fn=openai.Image.create_variation, 
            parser_fn=lambda response: response.data[0].url,
            size=size,
            image=image_bytes, 
            n=n, **kwargs)

        return result


    
    def get_user_limits(self, user_phone_number):
        #TODO: implement token limiting and message limiting
        raise NotImplementedError
        
    

# Flask app webhook handler for Twilio SMS    
app = Flask(__name__)

@app.route("/sms", methods=['POST'])
def incoming_sms():
    textgpt.handle_incoming_message(request.values)
    return ''

if __name__ == "__main__":
    textgpt = textGPT(TWILIO_PHONE_NUMBER)
    textgpt.mdb.add_system_prompt("Your role is to respond to texts like Eric Cartman from South Park.")
    

    app.run(debug=False, port=6969)