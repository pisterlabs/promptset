import docx
import openai

# Configura la clave de la API de OpenAI
openai.api_key = 'key'

# Abre el archivo de Word
doc = docx.Document('informe.docx')

# Lee el contenido del documento
document_text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])

# Envía una solicitud a la API para generar el resumen
response = openai.Completion.create(
    engine='text-davinci-003',
    prompt=document_text,
    max_tokens=150,
    temperature=0.3,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
)

summary = response.choices[0].text.strip()

# Imprime el resumen generado
print(summary)
