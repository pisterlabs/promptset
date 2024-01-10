# Importing the openai library and the os module
import openai
import os
# Loading environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()
# Setting the OpenAI API key to the value stored in the environment variable "OPENAI_API_KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Importing the requests library
import requests

# Defining a test URL to fetch an image
url_test = "https://oaidalleapiprodscus.blob.core.windows.net/private/org-2dA58464KfGXsNQqhZrf0VqG/user-USitdQFQhBbmQsgHPhbzMEA1/img-blnmD1xZiFnbLFS36Dh4kg6s.png?st=2023-02-04T17%3A52%3A18Z&se=2023-02-04T19%3A52%3A18Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-02-03T23%3A27%3A51Z&ske=2023-02-04T23%3A27%3A51Z&sks=b&skv=2021-08-06&sig=LhuKHpnknvEscnc%2BNUUh7XX2TBcPEo1nNN/AGWm/2zQ%3D"

# Defining a function that takes in an input text and returns the URL of the generated image
def return_url(input_text):
    # Making a request to the OpenAI API to generate an image based on the input text
    response = openai.Image.create(
    prompt = input_text,
    size = "512x512"
    )
    # Returning the URL of the generated image
    return str(response['data'][0]['url'])

# Defining a function that takes in a URL and a picture name and saves the image from the URL to the file system
def save_image(url, picture_name):
    # Fetching the image data from the URL
    img_data = requests.get(url).content
    # Saving the image data to a file with the specified picture name
    with open('graphics/' + str(picture_name) + '.jpg', 'wb') as handler:
        handler.write(img_data)
    # Returning a confirmation message that the image was saved
    return "image " + picture_name + " saved"

# Defining a function that takes in an input text and a picture name and generates an image based on the input text, then saves the image to the file system
def make_image_and_save(input_text, picture_name):
    # Getting the URL of the generated image based on the input text
    url = return_url(input_text)
 
    # Save the image from the URL
    save_image(url, picture_name)
    
#make_image_and_save("rustic computer", "rustyboi")
