import pandas as pd
import requests
import json
import openai



#Api
Eletronics_api_url = 'https://fakestoreapi.com/products/category/electronics'
def getAllEletronicsFromApi():
    response = requests.get(f'{Eletronics_api_url}')
    dataframe = pd.DataFrame(response.json())
    dataframe.to_csv('User_eletronics.csv')
    return dataframe
    

#CSV
def get_Name_ByCategory(category):
    df = pd.read_csv('User_costumer.csv')
    list = []
    for index, row in df.iterrows():
        if row['Category'] == category:
            list.append(row['Name'])
    return list

todosEletronicos = getAllEletronicsFromApi()
Usuarios = get_Name_ByCategory('eletronics')


openai.api_key = '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'

def generate_ai_news(user):
  completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
      {
          "role": "system",
          "content": "Você é um especialista em markting para Lojas."
      },
      {
          "role": "user",
          "content": f"Crie uma mensagem para {user} sobre a importância dos investimentos (máximo de 100 caracteres)"
      }
    ]
  )
  return completion.choices[0].message.content.strip('\"')

for user in Usuarios:
  news = generate_ai_news(user)
  print(news)
  user['news'].append({
      "icon": "https://digitalinnovationone.github.io/santander-dev-week-2023-api/icons/credit.svg",
      "description": news
  })
  
print(user['news'])


