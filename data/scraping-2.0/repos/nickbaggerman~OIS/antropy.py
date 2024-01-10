import sys
import re

# Maak de input een complete string aan tekst
string = ''
for word in sys.argv[1:]:
    string += word + ' '

def extract_content_text(chat_message):
  '''Zorgt ervoor dat ChatGPT output een mooie string wordt.'''
  # Define a regular expression pattern to match the content within 'content' lines
  pattern = r"content='(.*?)'"

  # Use re.search to find the match in the input string
  match = re.search(pattern, str(chat_message))

  # If a match is found, return the content text; otherwise, return None
  return match.group(1) if match else None

def replace_newline_with_br(input_string):
    output_string = input_string.replace('\n', '<br>')
    return output_string

# Start ChatGPT samenwerking
content = 'Ik ga je een vraag stellen, maar houd rekening met het volgende: Je bent een AI-chatbot genaamd Chat-Bob. Ik wil dat je antwoorden zo menselijk mogelijk zijn. Dat betekent geen opsommingen of codefragmenten, maar alleen natuurlijke zinnen die je in een gesprek kunt terugvinden. Geef antwoorden van maximaal ongeveer 100 woorden, maar geef nog steeds inzichtelijke antwoorden. Doe je best om het gesprek gaande te houden. Geef antwoorden die gemakkelijk te begrijpen zijn. Je mag me vragen stellen of twijfels met me bespreken om de discussie te stimuleren, maar doe dit natuurlijk met mate. Je antwoorden moeten geformatteerd zijn op een manier die geschikt is voor HTML. Hier is de vraag: '
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo-0301",
  messages=[
    {"role": "system", "content": content},
    {"role": "user", "content": string}
  ]
)

answer = completion.choices[0].message

print(replace_newline_with_br(extract_content_text(answer)))

"""
response = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt="Write a tagline for an ice cream shop."
)
"""