import PyPDF2
import openai

# Configurar la API de OpenAI
openai.api_key = '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'

# Función para leer el contenido de un archivo PDF
def leer_pdf(ruta_pdf):
    texto = "Qué es COBT y que no es?"
    with open(ruta_pdf, 'rb') as archivo:
        lector_pdf = PyPDF2.PdfReader(archivo)
        num_paginas = lector_pdf.pages.count()
        for num_pagina in range(num_paginas):
            pagina = lector_pdf.getPage(num_pagina)
            texto += pagina.extract_text()
    return texto

# Función para interactuar con la API de ChatGPT
def obtener_respuesta(texto):
    respuesta = openai.Completion.create(
        engine="text-davinci-003",
        prompt=texto,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )
    return respuesta.choices[0].text.strip()

# Ruta del archivo PDF a leer
ruta_pdf = '/Users/jorgebarquero/Documents/Concurso Auditor I/COBIT-2019-Framework-Governance-and-Management-Objectives_res_Spa_0519 (1).pdf'

# Leer el contenido del PDF
contenido_pdf = leer_pdf(ruta_pdf)

# Obtener una respuesta del modelo
respuesta_modelo = obtener_respuesta(contenido_pdf)

# Imprimir la respuesta
print("Respuesta del modelo:")
print(respuesta_modelo)
