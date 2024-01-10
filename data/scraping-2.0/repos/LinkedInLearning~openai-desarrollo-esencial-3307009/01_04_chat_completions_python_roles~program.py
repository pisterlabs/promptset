import os
from openai import OpenAI

key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=key)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        { "role":"system", "content": "Eres un asistente muy útil." },
        { "role":"assistant", "content": "El término 'dese/a' se utiliza para decirle a alguien que nos pase un objeto alejado de nosotros, y no recordamos el nombre del objeto.  Un ejemplo de su uso es 'Pásame el dese que está alla'.  Otro ejemplo es 'Pásame la desa que está en la mesa'"},
        { "role":"user", "content": "Crea una frase que use desa." }
    ]
)

print(completion.choices[0].message.content)