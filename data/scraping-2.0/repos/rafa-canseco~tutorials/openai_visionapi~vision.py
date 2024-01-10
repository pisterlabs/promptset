##Pip install openai
##pip install python-dotenv
from openai import OpenAI
from dotenv import load_dotenv
import base64
import os
load_dotenv()

# Importar la clave de la API desde un archivo .env para mantenerla segura
api_key = os.getenv("API_KEY")

# Crear una instancia de la clase OpenAI con la clave de la API
model = OpenAI(api_key=api_key)

# Definir una función para convertir una imagen a su representación en base64
def image_b64(image):
    # Abrir la imagen y codificarla en base64
    with open(f"images/{image}", "rb") as f:
        return base64.b64encode(f.read()).decode()

# Convertir la imagen 'image.png' a base64 para su envío
b64_image = image_b64("image.png")


# Realizar una petición al modelo de OpenAI, enviando la imagen en base64
response = model.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{b64_image}"
                },
                {
                    "type": "text",
                    "text": "what is in this image?"
                }
            ]
        }
    ],
    max_tokens=1024,
)

# Extraer la respuesta del modelo y mostrar el contenido de la misma
message = response.choices[0].message
message_text = message.content

# Imprimir el texto de la respuesta
print(message_text)