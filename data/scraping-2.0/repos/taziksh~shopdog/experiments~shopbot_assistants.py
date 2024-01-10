from re import search
from openai import OpenAI

client = OpenAI()

search_google_shopping_json = {
    "name": "search_google_shopping",
    "parameters": {
      "type": "object",
      "properties": {
        "product_type": {
          "type": "string"
        },
        "gender": {
          "type": "string",
          "enum": [
            "male",
            "female",
            "unisex"
          ]
        },
        "size": {
          "type": "string",
          "enum": [
            "XS",
            "S",
            "M",
            "L",
            "XL"
          ]
        },
        "price": {
          "type": "string"
        }
      },
      "required": [
        "product_type"
      ]
    },
    "description": "Searches Google Shopping to find URLs to the top 10 most relevant products"
  }

assistant = client.beta.assistants.create(
  instructions="You are a helpful assistant that searches Google Shopping to find URLs to the top 10 most relevant products",
  model="gpt-4-1106-preview",
  tools=[{"type": "function", "function": search_google_shopping_json}]
)

thread = client.beta.threads.create(
    messages=[
        {
            "role": "user",
            "content": "Men's white v-neck cardigan sweater XS size between $75 and $150"
        }
    ]
)

import pdb; pdb.set_trace()
run = client.beta.threads.runs.create(thread_id = thread.id, assistant_id = assistant.id)