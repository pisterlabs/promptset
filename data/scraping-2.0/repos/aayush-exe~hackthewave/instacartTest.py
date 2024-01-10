import os
import requests
import json
import openai

openai.api_key = "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"

# Step 1: Read and parse the manifest file
with open('instacartmanifest.json', 'r') as file:
    manifest_data = json.load(file)

# Extract the message from the manifest data
desc = manifest_data.get('description_for_model', '')
ingred = "steak, rosemary, butter, potato"
# Step 2: Make the API call
API_URL = "https://api.openai.com/v1/engines/davinci/completions"
HEADERS = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json",
}

data = {
    "prompt": desc + 
    
    '''

    openapi: 3.0.1
info:
  title: Instacart
  description: Order from your favorite local grocery stores.
  version: 'v2.1'
servers:
  - url: https://www.instacart.com
paths:
  /rest/llm_integration/openapi/v2_1/recipes:
    post:
      operationId: create
      summary: Create an Instacart link to the shopping list of ingredients.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/createRequest'
      responses:
        "200":
          description: Instacart link to the shopping list of ingredients.
        "400":
          description: Could not create an Instacart link to the shopping list of ingredients.
components:
  schemas:
    createRequest:
      type: object
      properties:
        title:
          type: string
          description: Recipe title (e.g. "Vanilla Yogurt Parfait")
          required: true
        ingredients:
          type: array
          items:
            type: string
          description: List of strings where each element is a recipe ingredient (e.g. ["2 cups of greek yogurt", "2 tablespoons of honey", "1 teaspoon of vanilla extract"]). Don't include items in the list that the user already mentioned they have.
          required: true
        instructions:
          type: array
          items:
            type: string
          description: List of strings where each element is a recipe instruction
          required: true
        question:
          type: string
          description: This field stores the question asked by the user about recipe or mealplan in the current chat session. For instance, a user can ask "recipe for chocolate cookies" and the assistant responds by listing the ingredients needed to make chocolate cookies. In this chat interaction, we need to return "recipe for chocolate cookies" as the value in this field
          required: true
        partner_name:
          type: string
          description: The value used to populate this field should always be "OpenAI"
          required: true
        ''' +

    "Now off of this data, create a link to an Instacart cart with these ingredients in it: " + ingred,
    "max_tokens": 150
}

response = requests.post(API_URL, headers=HEADERS, json=data)
response_data = response.json()

print(response_data.get('choices', [{}])[0].get('text', '').strip())

