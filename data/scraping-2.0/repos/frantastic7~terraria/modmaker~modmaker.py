import os
import re
import subprocess
from rembg import remove
from PIL import Image
from dotenv import load_dotenv
import openai
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

gpt_prompt = """

You are TModBot! An designed to help write Terraria mods. Your mission is to write C# files for custom Terraria swords.

Make a really cool name.  Be as creative as possible, no "test sword", "example sword"! These swords are going to be used for rich world building!

Each sword you make should be customized to your liking. Make some swords op and imba! Make some swords bad, make some small, some big, some fun, some not. You will also be the one coming up with an appropriate crafting recipe and name for your sword.

After your C# code, provide the crafting recipe for your sword. 

"""

message=[{"role": "user", "content": gpt_prompt}]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages = message,
    temperature=0.3,
    max_tokens=1000,
    frequency_penalty=0.3
)


match = re.search(r"```csharp([^`]*)```", response.choices[0].message.content)
if match:
    text = match.group(1)
else :
    match = re.search(r"```C#([^`]*)```", response.choices[0].message.content)
    if match :
        text = match.group(1)
    else : 
        match = re.search(r"using([^`]*)Crafting", response.choices[0].message.content)
        if match :
            text = "using"+match.group(1)
        else :
            print ("error, please try again")


sword_name = re.search(r"DisplayName.SetDefault([^`]*)Tooltip.SetDefault", response.choices[0].message.content)
name = sword_name.group(1)
item_name = name[2:].rstrip()[:-3]


subprocess.run(['touch',f'{item_name}.cs'])
with open (f"{item_name}.cs","w") as file:
    file.write(str(text))

crafting = re.search(r"Crafting ([^`]*)",response.choices[0].message.content)
crafting_recipe = "Crafting " + crafting.group(1).rstrip()

print (item_name)
print (crafting_recipe)


draw_prompt = f"Draw a pixel art sword, called {item_name}. Draw it diagonally from bottom left to top right. Draw it on a white background. Use only #ffffff for the background"

img_file = item_name + ".png"
dalle_image = openai.Image.create(
prompt = draw_prompt,
n=1,
size="256x256"
)

image_url = dalle_image['data'][0]['url']

subprocess.run(["curl","-o",img_file,image_url])

img = Image.open(img_file)

output = remove(img, treshold = 240)

output.save(img_file)

subprocess.run(["open",img_file])



