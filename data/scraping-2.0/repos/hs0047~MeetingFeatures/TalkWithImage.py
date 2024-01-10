import requests
from PIL import Image
import openai

# Set up OpenAI API credentials
openai.api_key = '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'

# Function to download the image from a URL and save it locally
def download_image_from_url(image_url, save_path):
    response = requests.get(image_url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

# Function to process the downloaded image and generate a response
def process_image(image_path):
    image = Image.open(image_path)
    # Your image processing code here...
    prompt = "Ask a question about the image:"
    return prompt

# Function to interact with ChatGPT and get a response
def chat_with_gpt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        temperature=0.6
    )
    answer = response.choices[0].text.strip()
    return answer

# Main code
image_url = "https://wallpapers.com/images/featured/random-people-nr23d7xjroka0x1y.jpg"  # Replace with the URL of your image
image_path = "sample_image.jpg"  # Local path to save the downloaded image

try:
    download_image_from_url(image_url, image_path)
    prompt = process_image(image_path)
except Exception as e:
    print(f"Error downloading or processing the image: {e}")
    exit()

print("Image downloaded and processed successfully.")
print("Prompt:", prompt)

while True:
    question = input("Ask a question (or 'exit' to end): ")
    if question == "exit":
        break
    response = chat_with_gpt(prompt + " " + question)
    print("ChatGPT:", response)

print("Conversation ended.")
