import gspread
import openai
import requests
import stylegan2
from PIL import Image, ImageDraw, ImageFont
import os

# Set up Google Sheets API
keyfile_path = 'path/to/your/keyfile.json'
scope = ['https://www.googleapis.com/auth/spreadsheets']
credentials = ServiceAccountCredentials.from_json_keyfile_name(keyfile_path, scope)
gc = gspread.authorize(credentials)

# Set up OpenAI API
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# Set up Instagram Graph API
access_token = 'YOUR_INSTAGRAM_ACCESS_TOKEN'
api_endpoint = 'https://graph.instagram.com/v12.0/me/media'

# Function to add text overlay
def add_text_overlay(image, text):
    draw = ImageDraw.Draw(image)
    position = (50, 50)  # Adjust the position as needed
    font = ImageFont.load_default()
    text_color = (255, 255, 255)  # White color
    draw.text(position, text, fill=text_color, font=font)
    return image

# Function to fetch prompts from Google Sheets
def get_prompts_from_google_sheets():
    worksheet = gc.open('Your Spreadsheet Name').sheet1
    prompts = worksheet.col_values(1)  # Assuming prompts are in the first column
    return prompts

# Function to generate text with ChatGPT
def generate_text_with_gpt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text

# Load the pre-trained StyleGAN2 model
network_pkl = 'your_pretrained_model.pkl'
_G, _D, Gs = stylegan2.run.load_networks(network_pkl)

# Generate and post images
if __name__ == "__main__":
    prompts = get_prompts_from_google_sheets()

    for prompt in prompts:
        generated_text = generate_text_with_gpt(prompt)

        # Generate an image using StyleGAN2
        latent_vector = stylegan2.run.generate_latent()
        image = Gs.run(latent_vector)

        # Add the generated text as an overlay
        image_with_overlay = add_text_overlay(image, generated_text)

        # Save the image with overlay
        image_with_overlay.save('image_with_overlay.png')

        # Post the image to Instagram's story
        data = {
            'image_url': 'URL_TO_YOUR_GENERATED_IMAGE',  # Replace with the image with overlay URL
            'caption': generated_text,
        }

        response = requests.post(
            api_endpoint,
            params={'access_token': access_token},
            data=data
        )

        if response.status_code == 200:
            print('Successfully posted to your Instagram story!')
        else:
            print('Error posting to your Instagram story:', response.status_code, response.text)