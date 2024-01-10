import pandas as pd
import requests
import openai
import time
teste='-'
openai.api_key = 'xxx' #Coloque a SUA KEY do CHATGPT
df = {0:{'name': 'João', 'entrada': None},
      1:{'name': 'Maria', 'entrada': None},
      2:{'name': 'Oswald', 'entrada': None},
      3:{'name': 'Arthur', 'entrada': None},
      4:{'name': 'Cebola sem Sal', 'entrada': None},}
print(teste.center(10, "-"))
print(df)
print(teste.center(10, "-"))
for i in range(4):
  time.sleep(3)
  gpt = openai.ChatCompletion.create(model="gpt-3.5-turbo",
      messages=[
      {"role": "system","content": "Você é um escritor de livros de genero de autoajuda"},
      {"role": "user","content": f"Crie uma mensagem para {df[i]['name']} com pensamentos positivos com no máximo 100 caracteres"}])
  
  print(teste.center(4, '-'))
  time.sleep(3)
  print(gpt.choices[0].message.content)
  time.sleep(3)
  print(teste.center(4, '-'))
  time.sleep(3)
  df[i]['entrada']= gpt.choices[0].message.content.strip('\"')
print(teste.center(10, "-"))
print(df)