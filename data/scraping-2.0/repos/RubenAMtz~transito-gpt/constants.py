import openai
import tiktoken
import os
import random

HEADER = """Las preguntas tratarán de describir una situación, y quizá vengan acompañadas de una pregunta. Contesta el comentario y/o pregunta apoyándote del contexto MAS relevante y de tu conocimiento general en el tema. Describe tu respuesta con un tono amigable y formal, trata de ponerte en los zapatos del usuario. Contesta desde el punto de vista de un conductor a menos que se indique lo contrario. Se claro y trata de evitar palabras ambiguas. Si la pregunta o comentario no tienen que ver con leyes de tránsito no respondas, repito, no contestes a preguntas o comentarios que no sean relacionadas a las Leyes de Tránsito.\n\nContexto:\n"""

HEADER = """Imagina que eres un abogado de leyes de tránsito. Un usuario te ha hecho una pregunta o comentario en el que te pide ayuda. Las preguntas tratarán de describir una situación, y quizá vengan acompañadas de una pregunta. Contesta el comentario y/o pregunta apoyándote del contexto MAS relevante y de tu conocimiento general en el tema. Si la pregunta o comentario están fuera del contexto de la ley de tránsito invita al usuario a reformular su pregunta y hazle saber porqué lo haces. Por ejemplo:

Pregunta: ¿Qué pasa si me detienen por manejar ebrio?

*Revisa las referencias y elige la respuesta más adecuada*
Respuesta: Según las leyes de tránsito ... etc.

Pregunta: ¿Puedo cruzarme un semáforo en amarillo?

*Revisa las referencias y elige la respuesta más adecuada*
Respuesta: La ley dice que ... etc.

\n\Referencias:\n
"""

QUESTIONS = [
    "¿Qué pasa si me detienen por manejar ebrio?",
    "¿Puedo cruzarme un semáforo en amarillo?",
    "¿Puedo estacionarme en doble fila?",
    "¿Puedo estacionarme en la banqueta?",
    "¿Me puedo estacionar en sentido contrario?",
    "¿Puedo conducir con comida en la mano?",
    "¿Puedo conducir con mi mascota?"
]

QUESTION = random.choice(QUESTIONS)

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

MAX_SECTION_LEN = 1000
SEPARATOR = "\n* "
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002
ENCODING_MODEL = 'p50k_base' # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

# try to load api key from file, otherwise use environment variable but do not raise error if not found
if os.path.exists("keys.txt"):
    openai.api_key_path = "keys.txt"
else:
    openai.api_key = os.environ.get("OPENAI_API_KEY")

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 500,
    "model": COMPLETIONS_MODEL,
}