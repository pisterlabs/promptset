import openai
import re
from api_key import API_KEY
openai.api_key = API_KEY

model_engine = "text-davinci-003"

text= input("What do you want to know about?")
prompt = text

print("AI-Bot is generating new text for you....")

completions = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop = None,
    temperature = 0.5
)
#print the generated text
generated_text = completions.choices[0].text

# save text in file
with open("generated_text.txt","w") as file:
    file.write(generated_text.strip())

print("text generated !!")