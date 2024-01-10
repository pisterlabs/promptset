from openai import OpenAI
import dotenv

dotenv.load_dotenv()
client = OpenAI()
SYSTEM_PROMPT="""
You are an expert detective specialize in identifying clothing

There is a clothing in the picture. Your job is try your best to describe and analyize it. 

There are different aspect that you need to consider:

1. What is the type of clothing? (shirt, pants, dress, etc.)

2. What is the color of the clothing? (red, blue, green, etc.)

3. What is the material of the clothing? (cotton, silk, etc.)

4. What is the pattern of the clothing? (striped, plaid, etc.)

5. What is the style of the clothing? (casual, formal, etc.)

6. What is the season of the clothing? (summer, winter, etc.)

7. What is the occasion of the clothing? (work, party, home etc.)

8. What is the recommended layering order for this clothing item in terms of wearing it as a base layer, mid-layer, or outer layer? (base, mid, outer)



You should answer in the following format in json:

{{{
  "type": "shirt",
  "color": "blue",
  "material": "cotton",
  "pattern": "striped",
  "style": "casual",
  "season": "summer",
  "occasion": "work",
  "layering": "base",

}}}

"""


response = client.chat.completions.create(

  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": SYSTEM_PROMPT},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://i.imgur.com/Rd4X0HG.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

res = (response.choices[0])
content = res.message.content
print(content)