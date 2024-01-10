import openai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

"""
## 5. Entregables

En esta sección se describen los entregables de la presente etapa que consisten en un 
script en Python para ChatGPT a través del API de OpenAI. Para ello se deberá crear las 
cuentas de plataforma y generar el API Key correspondiente (**NO SE DEBE INCLUIR LA API KEY**, 
revise https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety para ver 
como invocarle como variable de ambiente desde Python).


A. Diseña scripts y prompts que permitan:

    1) Obtener un resumen en español, de dos párrafos de longitud, del contenido del texto 
		`news_digital_bank.txt`. Adicionalmente, se deberá incluir un tercer párrafo que indique 
		cuál es el diario del que proviene el texto y el título correspondiente de la noticia. Este 
		programa se deberá guardar con nombre `conversacion_1.py`

    2) Crear 5 viñetas (bullets) que presenten los elementos más importantes de la historia el 
	    archivo `cuento.pdf` (usando el texto de todas las páginas del archivo). Adicionalmente, 
		se deberá incluir un par de viñetas que indique 1) el nombre del autor del texto, 2) los 
		personajes principales de la trama, 3) el título del cuento. Dicho programa se deberá guardar 
		con nombre `conversacion_2.py`

Cabe destacar que como resultado de los programas anteriores, se debe crear un script de 
conversación que guarde el script de conversación entre el usuario y ChatGPT en un archivo 
.txt (conversacion_i.txt donde i es el número de inciso asociado), con la estructura siguiente:

**Ejemplo**
```
Usuario: <Indicaciones del prompt empleado>

ChatGPT: Respuesta
```

Adicionalmente se deberá adjuntar capturar de pantalla en formato .png donde se aprecia 
el cuerpo de las conversaciones generadas por los ChatBots, se pueden usar numeraciones 
sucesivas sin son muchas fotos, ejemplo: evidencia_1_conversacion_i.png, 
evidencia_2_conversacion_i.png, ..., evidencia_5_conversacion_i.png
"""

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

### Extraccion de texto de un archivo .pdf ###

def reading_pdf(file_path: str) -> str:
    """
    Extrae el texto de un archivo .pdf
    
    Parameters;
        file_path (str): Ruta del archivo .pdf a analizar

    Salida:
        str: El texto extraído del archivo.
    """

    reader = PdfReader(file_path)
    page = reader.pages[0]
    return page.extract_text()

# Ruta del archivo .pdf a analizar
pdf_path_file = "./cuento.pdf"

# Extrae el texto del archivo .pdf
extracted_pdf = reading_pdf(pdf_path_file)

# print(extracted_pdf)
### Configuracion de la API de OpenAI ###

# Ejecuta una tarea de completar un texto
chat_completion = openai.chat.completions.create(
    model = "gpt-3.5-turbo",
    temperature = 0.5,
    messages = [
        {
            "role": "system",
            "content": "Actúa como experto en resumenes"
        },
        {
            "role": "user",
            "content": "Necesito que revises este texto:\n '''\n" + extracted_pdf + "\n'''\n Para esto requiero que hagas lo siguiente:\n 1) Crea 5 viñetas que presenten los elementos más importantes de la historia\n 2) En otro apartado, crea sub-viñetas que indique 1) el nombre del autor del texto, 2) los personajes principales de la trama, 3) el título del cuento."
        }
    ]
)
# Imprime el mensaje respuesta de ChatGPT
print(chat_completion.choices[0].message.content)
