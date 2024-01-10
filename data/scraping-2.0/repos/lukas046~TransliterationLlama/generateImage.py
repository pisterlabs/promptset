import openai
import urllib.request

# Get the api token
with open('token.txt') as f:
    lines = f.readlines()
openai.api_key = lines[1].strip()

# Generate the image


def generateImage(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="256x256",
    )

    # Save the image
    url = response["data"][0]["url"]
    urllib.request.urlretrieve(url, f"1.png")

# generateImage("tree")
