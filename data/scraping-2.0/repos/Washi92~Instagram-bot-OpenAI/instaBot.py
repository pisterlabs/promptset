import openai
import os
import urllib.request
from imageProcessing import overlay_text

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get('API_KEY')

# Set up your OpenAI API credentials
openai.api_key = API_KEY

async def generate_tip():
    # Prompt for generating tips and tricks
    prompt = "Generate a tip or trick for entrepreneurs:"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,  # Adjust the number of tokens as per your preference
        n=1,  # Generate a single completion
        stop=None,  # Let the model decide when to stop the completion
        temperature=0.7,  # Controls the randomness of the generated output
        top_p=1.0,  # Controls the diversity of the generated output
    )

    tip = response.choices[0].text.strip()
    return tip

# Generate a tip
#tip = generate_tip()
#print("Tip:", tip)


async def generate_quote():
    # Prompt for generating motivational quotes
    prompt = "Generate a motivational quote for entrepreneurs:"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,  # Adjust the number of tokens as per your preference
        n=1,  # Generate a single completion
        stop=None,  # Let the model decide when to stop the completion
        temperature=0.8,  # Controls the randomness of the generated output
        top_p=1.0,  # Controls the diversity of the generated output
    )

    quote = response.choices[0].text.strip()
    return quote

def save_image_from_url(url, file_path):
    try:
        urllib.request.urlretrieve(url, file_path)
        print("Image saved successfully!")
    except Exception as e:
        print(f"Error saving the image: {str(e)}")

async def motivate_me():
    # Generate a quote
    quote = generate_quote()
    print("Quote:", quote)

    # OpenAI request
    response = openai.Image.create(
    prompt="motivational landascape without text",
    n=1,
    size="1024x1024"
    )

    image_url = response['data'][0]['url']

    # SAVE THE IMAGE
    save_image_from_url(image_url, "MotivationalImg.jpg")
    # OVERLAY THE TEXT
    path = overlay_text("MotivationalImg.jpg", quote, ".\images\\")
    path = path.replace("\\","/")
    print(path)
    return path
    



