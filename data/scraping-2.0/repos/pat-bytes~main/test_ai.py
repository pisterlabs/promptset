import os
import openai

openai.api_key = ('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')

start = "Your are a AI Search Engine, answer the following query with a witty answer and include validated facts only."

def generate(prompt):
    start_sequence = "{}.{}".format(start,prompt)
    print("Requête générée :", start_sequence)
    completions = openai.Completion.create(
      model="text-davinci-003",
      prompt=start_sequence,
      temperature=0.1,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0.51,
      presence_penalty=0.5,
      #stream = False,
      #echo = True
    )
          
    
    message = completions.choices[0].text.strip()
    print("Réponse générée :", message)
    if message:
        print("reponse : ",message )
    else:
        print("vide")  
    
    return message