import requests
import json
import openai
import os
from dotenv import load_dotenv
load_dotenv()


url = "https://api.openai.com/v1/chat/completions"

# openai.api_key = os.getenv("OPENAI_API_KEY")

message = """
Message : ‎Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.

Message : ‎~ Nirant created this group

Message : ‎This group was added to the community “Generative AI”

Message : ‎Anyone in the community “Generative AI” can request to join this group by messaging group admins.

Message : ‎<attached: 00000012-PHOTO-2023-04-14-20-37-09.jpg>

Message : Has anyone used LangChain with Azure endpoints instead of OpenAI  directly ?

Message : https://python.langchain.com/en/latest/modules/models/llms/integrations/azure_openai_example.html
Quoted Message : Has anyone used LangChain with Azure endpoints instead of OpenAI  directly ?

Message : Tried this?

Message : Yup I’ve tried this 
It’s working for simple examples

Im looking to implement agents and not able to find documentation for it
Quoted Message : Tried this?

Message : They have examples for custom LLM agents, not sure if that helps
"""

payload = {
    "model": "gpt-3.5-turbo-0613",
    "messages": [
        {
            "role": "user",
            "content": message,
        }
    ],
    "functions": [
        {
            "name": "classify_message_block",
            "description": "This function classifies a given message block into one of the given types",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": ["langchain", "engineering"]
                    },
                },
                "required": ["topic"]
            }
        }
    ]
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f'Bearer ${os.getenv("OPENAI_API_KEY")}',
}

response = requests.post(url, data=json.dumps(payload), headers=headers)

if response.status_code == 200:
    result = response.json()
    # do something with the result
    import pdb; pdb.set_trace()
else:
    print("Error:", response.status_code, response.text)

