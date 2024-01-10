import openai 
import urllib 

openai.api_key = ''


response = openai.Image.create(
    prompt = "A futuristic city at night",
    n=1,
    size="512x512"
)


image_url = response['data'][0]['url']
urllib.request.urlretrieve(image_url, 'test.jpg')