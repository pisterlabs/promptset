import openai
import sys

openai.api_key = "YOUR_API_KEY"

if len(sys.argv) != 2:
    print("Uso: python gpt_imprimible.py <prompt>")
    sys.exit(1)

prompt = sys.argv[1]

formatted_prompt = f"¿Es {prompt} algo imprimible en 3D o no? Responde con una sola palabra (Sí./No.)"

def preguntar_a_chatgpt(pregunta):
    respuesta = openai.Completion.create(
        engine="text-davinci-002",
        prompt=pregunta,
        max_tokens=50
    )
    return respuesta.choices[0].text.strip()

respuesta_gpt = preguntar_a_chatgpt(formatted_prompt)

# Imprimo la respuesta para que genie.py la capture
print(respuesta_gpt)
