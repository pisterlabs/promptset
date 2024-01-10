import openai
import json
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API")

prompt = ''' Imagine you are an AI Slides generating tool called 'SlidesGenie'. Based on the input of the user, create the structure of the slides presentation as a JSON object.
You have 4 slide formats at your disposal to use. Each slide has a type_id and takes different inputs. Slides with images have a special input called image_prompt. This should be a description of an image that can be generated using a text-to-image model. The description should match with the contents of the slide. The slide formats are:
1. Title Slide
The title slide consists of a title and a subtitle.
type_id:title
inputs: title,subtitle
2. Slide with Image on left
The slide consists of an image on the left, a title on the right, and a body of text under the title.
type_id: left-image-text
inputs: title, image_prompt, body.
3. Slide with Image on Right
The slide consists of an image on the right, a title on the left, and a body of text under the title.
type_id: right-image-text
inputs: title, image_prompt, body.
4. Slide with only text
The slide consists of a title, subtitle and body
type_id: title-sub-text
inputs: title, subtitle, body

Template:
```
{
  "slides": [
    {
      "type_id": "title",
      "inputs": {
        "title": "<insert-title>",
        "subtitle": "<insert-subtitle-here>"
      }
    },
    {
      "type_id": "left-image-text",
      "inputs": {
        "title": "<insert-title>",
        "image_prompt": "<insert-image-generating-prompt>",
        "body": "<insert-body>"
      }
    },
    {
      "type_id": "right-image-text",
      "inputs": {
        "title": "<insert-title>",
        "image_prompt": "<insert-image-generating-prompt>",
        "body": "<insert-body>"
      }
    },
    {
      "type_id": "title-sub-text",
      "inputs": {
        "title": "<insert-title>",
        "subtitle": "<insert-subtitle>",
        "body": "<insert-body>"
      }
    }
  ]
}
```
The above JSON contains a list of the 4 slide templates: title, left-image-text, right-image-text and title-sub-text. Use this to create 10 slides. For each slide, pick an appropriate slide template from the 4 templates given in the JSON and generate the response. Be creative and factual with the content. Comply with the user's input. 
RESPOND WITH JSON ONLY

'''
def content_generation(user_input):
  user_input = "user input : "+ user_input

  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
              {"role": "system", "content": prompt},
              {"role": "user", "content": user_input},
          ]
  )

# print(response["choices"][0]["message"]["content"])

  d = json.loads(response["choices"][0]["message"]["content"]) 
  print(d)

  return d
  
#presentation format