import openai
import urllib

API_KEY = "YOUR_API_KEY"
client = openai.OpenAI(api_key=API_KEY)

response = client.images.generate( # 생성 
    model = "dall-e-2", 
    prompt = "Sunset wearing synthwave-style crown reflecting over the sea, digital art", 
    size = "1024x1024",
    quality = "standard",
    n=1, 
)

image_url = response.data[0].url
urllib.request.urlretrieve(image_url, "image.jpg")
print(image_url)
