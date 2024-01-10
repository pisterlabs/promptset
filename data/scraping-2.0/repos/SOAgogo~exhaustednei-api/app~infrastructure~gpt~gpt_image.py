from openai import OpenAI
from dotenv import load_dotenv
import sys
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

if len(sys.argv) == 2:
    image_path = sys.argv[1]
    # image_path = 'https://www.niusnews.com/upload/imgs/default/202207_Jennie/0701cat/03.JPG'
    response = client.chat.completions.create(
      model="gpt-4-vision-preview",
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": f"According to the dog or cat species, give me some information about how to take care of this animal, and the information must be related to the animal species, the words must be less than 70 words."},
            {
              "type": "image_url",
              "image_url": {
                "url": image_path,
              },
            },
          ],
        }
      ],
      max_tokens=4096,
    )
    print(response.choices[0])

    
else:
    print("No message received from Ruby.")




