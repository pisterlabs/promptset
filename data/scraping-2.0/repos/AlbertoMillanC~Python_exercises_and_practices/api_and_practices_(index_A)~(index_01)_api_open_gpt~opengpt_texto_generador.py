import openai

# Lee la API Key desde el archivo "api_key.txt"
with open("./api_key.txt", "r") as file:
    api_key = file.read()
# Inicializa el modelo GPT-3
openai.api_key = api_key
model_engine = "text-davinci-002"

# Define el prompt para generar el texto
prompt = "Genera un texto en prosa para una obra infantil sobre un pequeño zorro con contenido para niños, en palabras sencillas, lenguaje colombiano y de forma de parrafos facil de leer. "

# Genera el texto utilizando el modelo GPT-3
text = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
).choices[0].text

# Imprime el texto generado
print(text)
