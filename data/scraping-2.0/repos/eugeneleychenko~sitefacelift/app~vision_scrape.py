
from openai import OpenAI
import base64
from dotenv import load_dotenv
import os
import json

load_dotenv()

# openai_api_key=os.getenv("OPENAI_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_path = "./gio-law.com_ (1).png"
base64_image = encode_image(image_path)

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
                  const testimonalData = (Return 3 testimonials from this site. They should be in an array of objects, including text, author, location. For example "[
  {
    text: `""Carmen was absolutely wonderful to work with. He was truly honest and I never felt taken advantage of. I can't recommend this law office and Carmen enough.""`,
    author: ""Jaime Oliver"",
    location: ""New York"",
  },
  {
    text: `""Even after the case we still keep in contact for any question that we still might have, for people who do not speak English I recommend him, he makes sure that the person in the case understands everything that happens in their case.""`,
    author: ""Atriz R"",
    location: ""Manhattan, New York"",
  }]")
                    
                    
              Use actual words examples from this mockup. If it is not available, say so.  no need for any full sentences.
                    """
                     
                },
                
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ],
        }
    ],
    max_tokens=500,
)

response_dict = json.loads(response.json())
print(response_dict['choices'][0]['message']['content'])

