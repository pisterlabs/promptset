from channels.generic.websocket import AsyncWebsocketConsumer
from .bot_with_conv_bob_backup import inputdata
from channels.generic.websocket import AsyncWebsocketConsumer , SyncConsumer
from channels.exceptions import StopConsumer
from time import sleep
import asyncio
import time   
from tenacity import retry, stop_after_attempt, wait_fixed
import requests
import re
from openai.error import APIError, APIConnectionError, RateLimitError
import openai
import json
conversation = []  # Define conversation as a global variable

class MySyncConsumer(SyncConsumer):
    # global conversation
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = requests.Session()
        
        
    def websocket_connect(self, event):
        print('Websocket Connected...', event)
        self.send({'type': 'websocket.accept'})
    
    def websocket_receive(self, event):
        print('Message Received from Client....', event['text'])
        
        question = event['text']
        try:
            
            response = inputdata(event['text'], conversation)
        except openai.error.APIError as e:
            # Pass the API error message to the template
            error_message = f"OpenAI API returned an API Error: {e}"
            print(error_message)
            self.send({'type': 'websocket.send','text': json.dumps({"error":error_message})})
            return
                   

        except openai.error.APIConnectionError as e:
        # Pass the connection error message to the template
            error_message = f"Failed to connect to OpenAI API: {e}"
            print(error_message)
            self.send({'type': 'websocket.send','text': json.dumps({"error":error_message})})
            return

        except openai.error.RateLimitError as e:
        # Pass the rate limit error message to the template
            error_message = "OpenAI API request exceeded rate limit"
            print(error_message)
            self.send({'type': 'websocket.send','text': json.dumps({"error":error_message})})
            return
        except Exception as e:
            # Handle other exceptions
            error_message = f"An error occurred: {e}"
            print(error_message)
            self.send({'type': 'websocket.send','text': json.dumps({"error":error_message})})
            return
        
        chunks = []
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
        def send_chunk(chunk):
            self.send({'type': 'websocket.send', 'text': json.dumps({"chunk":chunk})})
        # stop_generating = False
        for line in response:
            # print(line)
            if line['choices']:
                
                chunk = line['choices'][0].get('delta', {}).get('content', '')
                if chunk:
                    chunks.append(chunk)
                    try:
                        # print(chunk)
                        send_chunk(chunk)
                    except requests.exceptions.ChunkedEncodingError:
                        print('ChunkedEncodingError occurred. Retrying...')
                        continue
                time.sleep(0.1)

        # Send an indication that all chunks have been sent
        self.send({'type': 'websocket.send', 'text': json.dumps({"chunk":'ALL_CHUNKS_SENT'})})
        # print(chunks)
        answer = "".join(chunks)
        conversation.append({"question": question, "answer": answer})
        print(conversation)

    def websocket_disconnect(self, event):
        global conversation
        print('Websocket Disconnected...')
        conversation = []
        