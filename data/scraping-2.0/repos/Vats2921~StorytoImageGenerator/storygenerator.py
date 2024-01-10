import openai 

# Set API key
api_key = 'Enter_Your_api_key'
openai.api_key = api_key

# Generate story 
image_prompt = "Enter_story_prompt"
story_prompt = f"Generate a short story based on the image: '{image_prompt}'"
response = openai.Completion.create(
  engine="text-davinci-002",
  prompt=story_prompt,
  max_tokens=100
)
story = response.choices[0].text

# Generate image from story
image_prompt = f"Generate an image based on this short story: {story}" 
image_response = openai.Image.create(
  prompt=image_prompt,
  n=1,
  size="1024x1024"
)
image_url = image_response['data'][0]['url']

# Download image
import requests 
image_data = requests.get(image_url).content
with open('story.png', 'wb') as f:
  f.write(image_data)
  
print("Generated image from story saved to 'story.png'!")
