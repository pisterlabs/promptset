import openai
import json
import re
from dotenv import dotenv_values
from flask import Flask, render_template, request
from flask_talisman import Talisman

config = dotenv_values('.env')
openai.api_key = config['OPENAI_API_KEY']

application = Flask(__name__,   
            template_folder='templates',
            static_url_path='',
            static_folder='static'
            )

app = application

csp = {
    'default-src': [
        '\'self\'',
        'cdnjs.cloudflare.com'
    ]
}
#Talisman(application, content_security_policy=csp)

def generate_palette(text):
    messages = [
    {
        "role": "user",
        "content": "You are a color palette generating assistant. I'll give you a theme, mood, or context, and you'll provide a color palette. Here's an example: 'Tropical Beach'."
    },
    {
        "role": "assistant",
        "content": '["#D0E6A5", "#FFDD94", "#FA897B", "#CCABD8", "#3A5A40"]'
    },
    {
        "role": "user",
        "content": f"Convert the following verbal description of a color palette into a list of colors: {text}. Remember, the palette should STRICTLY contain between 2 and 8 hexadecimal color codes. No exceptions."
    }
]

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=500
)
    
    response_text = response.choices[0].message['content']
    print(response_text)
    #palette = json.loads(response_text.strip()) //Deprecated, Use Regex instead

    # Extract hexadecimal color codes from the response
    hex_codes = re.findall(r'(#(?:[A-Fa-f0-9]{6}|[A-Fa-f0-9]{3}))', response_text) # Regex to extract hex codes from the response

    # Post-processing
    if len(hex_codes) > 8:
        hex_codes = hex_codes[:8]  # Truncate to the first 8 colors if there are more than 8
    
    # Check the number of hex codes
    if 2 <= len(hex_codes) <= 8:  # If the number of hex codes is between 2 and 8
        print(hex_codes)
        return hex_codes

    # If not a valid number of hex codes, assume it's a message and return it
    else:
        print("Received a message instead of hex codes:", response_text)
        return response_text
    
    return hex_codes

def generate_description(initial_prompt, palette):
    messages = [
    {
        "role": "user",
        "content": f"I have a color palette {palette} generated for the theme '{initial_prompt}'. Can you provide a brief textual description (not more than two sentences) based on this palette and theme?"
    }
]

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=150
)
    
    return response.choices[0].message['content']
@application.route('/palette', methods=['POST'])
def prompt_for_palette():
    query = request.form.get("query")
    result = generate_palette(query)

    if isinstance(result, list):  # If the result is a list of hex codes
        palette = result
        description = generate_description(query, palette)
        return {"palette": palette, "description": description}
    
    else:  # If the result is a message
        return {"message": result}

@application.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    application.run(debug=True)