import os
import openai
import hashlib

cache_dir = "/tmp/gpt"

openai.api_type = "azure"
openai.api_base = "https://roc3demo.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.environ.get("OPENAI_API_KEY")



def callChatWithCache(messages):
    
    file_name = hashlib.md5(str(messages).encode()).hexdigest() + ".txt"
    file_path = os.path.join(cache_dir, file_name)
    if os.path.isfile(file_path):
      with open(file_path, "r") as f:
        result = f.read()
      
      return result

    response = openai.ChatCompletion.create(
      engine="hackyeah",
      messages = messages,
      temperature=0.7,
      max_tokens=800,
      top_p=0.95,
      frequency_penalty=0,
      presence_penalty=0,
      stop=None)
    
    with open(file_path, "w") as f:
      f.write("" + response.choices[0].message.content)
    return response.choices[0].message.content
    
#q = "Jestem uzdolniony manualnie i lubie samochody. Odpowiedz w dwu zdaniach."
q = "Chcę pracować w zespole F1"

responseKajtek = callChatWithCache([
      {"role":"system","content":"Jesteś kolegą, który pomaga dobrać przyszły zawód."},
      {"role":"user","content":q},
      ]);

print(responseKajtek)
print("=============================================")


response = callChatWithCache(
   [
      {"role":"system","content":"Jesteś koleżanką, która wraz z kolegą pomaga wybrać szkołę lub uniwesytet. Uzupełnij wypowiedź kolegi informacjami o szkołach lub uniwerystetach."},
      {"role": "assistant", "content": responseKajtek},
      {"role":"user","content": q},
      ]
)

# openai.ChatCompletion.create(
#   engine="hackyeah",
#   messages = [
#       {"role":"system","content":"Jesteś koleżanką, która wraz z kolegą pomaga wybrać szkołę lub uniwesytet. Uzupełnij wypowiedź kolegi informacjami o szkołach lub uniwerystetach."},
#       {"role": "assistant", "content": responseKajtek.choices[0].message.content},
#       {"role":"user","content": q},
#       ],
#   temperature=0.7,
#   max_tokens=800,
#   top_p=0.95,
#   frequency_penalty=0,
#   presence_penalty=0,
#   stop=None)

print(response)
