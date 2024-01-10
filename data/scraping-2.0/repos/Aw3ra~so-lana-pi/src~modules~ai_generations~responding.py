import openai
import os
import json

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')
file_path = os.path.join(src_path, 'conversation.json')

# Load in the conversation json file
with open(file_path) as json_file:
  conversation = json.load(json_file)


functions=[
      {
        "name": "crypto_send_money",
        "description": "Send some money to someone else",
        "parameters": {
            "type": "object",
            "properties": {
                "name":{
                    "type": "string",
                    "description": "The name of the person to send the tip to"
                },
                "amount":{
                    "type": "number",
                    "description": "The amount of money to send to the user",
                },
                "currency":{
                    "type": "string",
                    "description": "The currency to send the money in",
                },
            }
        },
        "required": ["name", "amount"]
      },
      {
        "name": "user_add_contact",
        "description": "Add a contact to the user's contacts",
        "parameters": {
            "type": "object",
            "properties": {
                "name":{
                    "type": "string",
                    "description": "The name of the contact to add"
                },
                "public_key":{
                    "type": "string",
                    "description": "The public key of the contact to add",
                },
            }
        },
      },
      {
        "name": "crypto_create_wallet",
        "description": "Create a wallet for the user",
        "parameters": {
            "type": "object",
            "properties": {
                "name":{
                    "type": "string",
                    "description": "The name of the wallet to create"
                },
            }
        },
      },
      {
        "name": "crypto_check_balance",
        "description": "Check the balance of a requested contact or wallet",
        "parameters": {
            "type": "object",
            "properties": {
                "name":{
                    "type": "string",
                    "description": "The name of the contact or wallet to check the transactions of",
                },
                "token":{
                    "type": "string",
                    "description": "The token to check the balance of",
                },
            }
        }
      },
      {
        "name": "crypto_check_transactions",
        "description": "Check the transactions of a requested contact or wallet, the name of the user is required",
        "parameters": {
            "type": "object",
            "properties": {
                "name":{
                    "type": "string",
                    "description": "The name of the contact or wallet to check the transactions of",
                },
                "interacted_with":{
                    "type": "string",
                    "description": "The name of the contact that the requested name has interacted with",
                },
                "amount":{
                    "type": "number",
                    "description": "The amount of transactions to return",
                    "default": "100",
                },
            }
        }, 
        "required": ["name"]
      },
      {
        "name": "qr_scan",
        "description": "Scan a QR code and return the data",
        "parameters": {
            "type": "object",
            "properties": {
                "name":{
                    "type": "string",
                    "description": "The name of the contact to scan the QR code of"
                },
            }
        },
      },
      {
        "name": "crypto_check_price",
        "description": "Check the price of a token",
        "parameters": {
            "type": "object",
            "properties": {
                "token":{
                    "type": "string",
                    "description": "The token to check the price of"
                },
            }
        },
      }
]


def generate_response(prompt):
  # Create a retry loop
  # Add the prompt to the conversation
  conversation.append({'role': 'user', 'content':'Awera: '+prompt+'.'})
  openai.api_key = os.environ["OPEN_AI_KEY"]
  while True:
    try:
        response = openai.ChatCompletion.create(
          model='gpt-4',
          messages=conversation,
          functions=functions,
        )
        # Save the conversation back to the json file, adding correct indentation
        if response.choices[0].finish_reason == 'function_call':
          retuend_response = response['choices'][0]['message']['function_call']
          return retuend_response
        else:
          conversation.append(response['choices'][0]['message'])
          with open(file_path, 'w') as outfile:
            json.dump(conversation, outfile, indent=4)
        return response['choices'][0]['message']['content'].replace("Lana: ", "")
    except Exception as e:
      print(e)
      print("Retrying...")
      continue



def generate_text(prompt):
  while True:
    try:
      openai.api_key = os.environ["OPEN_AI_KEY"]
      # If the content starts with {INFORMATION} or {ERROR}, then it is a command and dont append the user name to the prompt
      if prompt.startswith("{INFORMATION}") or prompt.startswith("{ERROR}"):
        conversation.append({'role': 'user', 'content':prompt+'.'})
      else:
        conversation.append({'role': 'user', 'content':'Awera: '+prompt+'.'})
      response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=conversation,
      )
      conversation.append(response['choices'][0]['message'])
      with open(file_path, 'w') as outfile:
        json.dump(conversation, outfile, indent=4)
      return response['choices'][0]['message']['content'].replace("Lana: ", "")
    except Exception as e:
      print(e)
      print("Retrying...")
      continue


if __name__ == "__main__":
  print(generate_response("Hello"))