import os
import openai
openai.api_key = "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"


text = input("Donner le travail ")

c =  openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
  {f"role": "user", "content": 'Donner moi une approximation  d\' intervalle de salaire de '+text+' en tunisie par mois, nombres seulement en reponse'}
    ]
)

Respone  = c["choices"][0]["message"]["content"]

Response = Respone.split()
salaries = []

for r in Response:
    try:
        s = float(r)
        salaries.append(s)
    except:
        continue
  
try:
    print("Salaire min : ",salaries[0], "  Salaire max : ",salaries[1])
except:
    print("No Salary")