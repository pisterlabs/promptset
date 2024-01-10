import os
import openai
from dotenv import load_dotenv
import spacy

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key

model = 'text-davinci-002'
prompt = "Cuenta una historia breve sobre un viaje a un país extranjero." 

response = openai.Completion.create(
  engine=model,
  prompt=prompt,
  n=1,
  max_tokens=100,
)

text = response['choices'][0]['text'].strip()

print (text)
print ("***")

model_spacy = spacy.load('es_core_news_md')
analyzed_text = model_spacy(text)

#for token in analyzed_text:
#  print(token.text, token.pos_, token.head.text, token.dep_)

#for token in analyzed_text.ents:
#  print(token.text, token.label_)

location = None

for ent in analyzed_text.ents:
  if ent.label_ == 'LOC':
    location = ent
    break

if location:
  prompt2 = f"¿Cuál es la capital de {location.text}?"
  response2 = openai.Completion.create(
    engine=model,
    prompt=prompt2,
    n=1,
    max_tokens=100,
  )

  text2 = response2['choices'][0]['text'].strip()

  print (text2)