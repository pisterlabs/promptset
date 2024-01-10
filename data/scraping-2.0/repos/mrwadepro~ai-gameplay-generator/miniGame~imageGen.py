from openai import OpenAI
import base64
client = OpenAI()


def getImage(prompt, filename):
    response = client.images.generate(
    model="dall-e-3",
    prompt= prompt,
    n=1,
    size="1024x1024",
    response_format="b64_json"
    ) 
    imgdata = base64.b64decode(response.data[0].b64_json)
    #filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)


getImage("Please generate a pixel art image of a button suitable for a farming game. This button will be used for player actions so it should be eye-catching and inviting. The dimensions should be 80 pixels wide by 60 pixels tall.", "button.jpg")