import openai
import asyncio
import hikari
import lightbulb
import os

API_KEY = os.environ.get('API_KEY')

def davinci_call(content):
            
            openai.api_key = "YOUR TOKEN HERE"
            return openai.Completion.create(
            model="text-davinci-003",
            prompt=content,
            max_tokens= 200,
            temperature=0
            )

def chat_call(context):
            openai.api_key = API_KEY
            return openai.ChatCompletion.create(
            model="gpt-4",
            messages = context,
            max_tokens= 450,
            temperature=0
            )

def chat_call_4(context):
            openai.api_key = "YOUR TOKEN HERE"
            return openai.ChatCompletion.create(
            model="gpt-4",
            messages = context,
            functions = [
                    {
      "name": "follow_up",
      "description": "returns true depending on whether a question is a follow up directly to the model beyond reasonable doubt",
      "parameters": {
        "type": "object",
        "properties": {
          "asked_follow_up": {
            "type": "string",
            "description": "Returns 'True' if the message was a follow up question to the last response, otherwise 'False'"
          }
        },
        "required": ["asked_follow_up"]
      }
    },
    {
      "name": "create_text_channel",
      "description": "Create a Discord Text Channel",
      "parameters": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "The channels name. Must be between 2 and 1000 characters."
          },
          "category":{
            "type": "string",
            "enum": ["text", "voice"],
            "description": "The category to create the channel under. This may be the object or the ID of an existing category. Return 'None' if not specified"
            
          }
        },
        "required": ["name, category"]
      }
    }
  ],
            max_tokens= 300,
            temperature=0
            )




