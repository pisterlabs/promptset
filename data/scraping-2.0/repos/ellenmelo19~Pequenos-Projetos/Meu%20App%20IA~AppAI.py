import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

#Executa até que o usuário digite 'exit'
while True:
  
  pergunta = input("\033[34mWhat is your question?\n\033[0m")
  
  # Verifica se o usuário digitou 'exit'
  if pergunta.lower() == "exit":
    # Fim do loop e mensagem de despedida
    print("\033[31mGoodbye!\033[0m")
    break
  
  # Resposta
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[
      {"role": "system", "content": "You are a helpful assistant. Answer the given question."}, 
      {"role": "user", "content": pergunta}
    ]
  )

  # Imprime a resposta
  print("\033[32m" + completion.choices[0].message.content + "\n")