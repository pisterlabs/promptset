
from openai import OpenAI
import json

quantity = "two"
OPENAI_API_KEY=""

client = OpenAI(api_key=OPENAI_API_KEY)

completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={ "type": "json_object" },
        messages=[
          {"role": "system", 
           "content": f'''You are a fashion expert specialized in clothing design. You will receive a description of an item of clothing, including type, color, style, and gender. Additionally, the description includes the ambient temperature. Your task is to provide {quantity} sets of clothing that include the item and fit the main description, without being modified. Each set should be represented as a JSON object, with attributes for "top", "bottom", and "shoes". Make sure each attribute is a string that describes the corresponding item of clothing in the format of: color item of clothing. The response must be a JSON object with an outfits element that is an array of the generated objects.'''},
          {"role": "user", 
           "content": "Man Casual Blue T-Shirt for summer in 25 degrees Celsius"}
        ]
    )

rpta = completion.choices[0].message

outfits_array = json.loads(rpta.content)

# guardar array en un archivo json
with open('outfits.json', 'w') as json_file:
    json.dump(outfits_array, json_file)