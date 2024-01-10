import seaborn as sns
from PIL import Image, ImageDraw

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

mood_description = input("Describe a mood or situation you want a mood for (Eg: colour like a blue sky at dusk) : ")
mood_prompt = f"The CSS code for a color like {mood_description}:\n\nbackground-color: #"

response = openai.Completion.create(
  model="text-davinci-003",
  prompt=mood_prompt,
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=[";"]
)

palette = f"#{response['choices'][0]['text']}"

width_px=100
new = Image.new(mode="RGB", size=(width_px,100))

newt = Image.new(mode="RGB", size=(100,100), color=palette)
new.paste(newt)

new.show()