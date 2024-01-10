import os
from openai import OpenAI

key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=key)

completion = client.chat.completions.create(
    max_tokens=1000,
    model="gpt-3.5-turbo-1106",
    messages=[
        { "role":"system", "content": "Eres un asistente muy útil." },
        { "role":"user", "content": 
         """
           Regresa todos los países del continente americano en JSON.
            Deben estar organizados en Norte América, Centro América y Sur América.
            No debe faltar ningún país.
            Por cada país, agrega las siguientes propiedades:
             1) Una que indique su población total aproximada según
                tus datos más recientes.
             2) Otra que indique su fecha de fundación o independencia.
                Si no tienes información al respecto, regresa 'Desconocida'.
         """ 
        }
    ], response_format={
        "type":"json_object"
    }
)

print(completion.choices[0].message.content)