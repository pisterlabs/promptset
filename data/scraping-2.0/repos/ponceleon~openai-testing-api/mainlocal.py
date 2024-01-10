################################################################################
### Step 1
################################################################################

# from flask import Flask, jsonify, request
import os
import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

# app = Flask(__name__)


# @app.route('/')
# def index():
#     form_html = """
#         <!DOCTYPE html>
#     <html>
#     <head>
#         <title>Prueba piloto de integración con OpenAI para modelos de conocimiento personalizados.</title>
#         <script src="https://cdn.tailwindcss.com"></script>
#             <style>
#         body {
#             margin: 0;
#             padding: 0;
#             box-sizing: border-box;
#         }
#         .center {
#             display: flex;
#             justify-content: center;
#             align-items: center;
#         }
#         .wave {
#             width: 5px;
#             height: 20px;
#             background: linear-gradient(45deg, cyan, #fff);
#             margin: 10px;
#             animation: wave 1s linear infinite;
#             border-radius: 20px;
#         }
#         .wave:nth-child(2) {
#             animation-delay: 0.1s;
#         }
#         .wave:nth-child(3) {
#             animation-delay: 0.2s;
#         }
#         .wave:nth-child(4) {
#             animation-delay: 0.3s;
#         }
#         .wave:nth-child(5) {
#             animation-delay: 0.4s;
#         }
#         .wave:nth-child(6) {
#             animation-delay: 0.5s;
#         }
#         .wave:nth-child(7) {
#             animation-delay: 0.6s;
#         }
#         .wave:nth-child(8) {
#             animation-delay: 0.7s;
#         }
#         .wave:nth-child(9) {
#             animation-delay: 0.8s;
#         }
#         .wave:nth-child(10) {
#             animation-delay: 0.9s;
#         }

#         @keyframes wave {
#             0% {
#                 transform: scale(0);
#             }
#             50% {
#                 transform: scale(1);
#             }
#             100% {
#                 transform: scale(0);
#             }
#         }
#     </style>
#     </head>
#     <body>
#         <div class="bg-gray-950 min-h-screen flex flex-col justify-center items-center">
#             <div class="flex items-center p-2">
#                 <h1 class="inline-block text-2xl sm:text-3xl text-center font-extrabold text-gray-100 tracking-tight">Preguntame sobre la <br>
#                     <a href="https://www.funcionpublica.gov.co/eva/gestornormativo/norma_pdf.php?i=186812" target="_blank"> LEY 2208 DEL 17 DE MAYO DE 2022 (CO) </a>
                    
#                 </h1>
#             </div>
#             <form class="w-full max-w-3xl bg-slate-800 p-8 rounded-lg">
#             <div class="flex flex-wrap -mx-3 mb-6">
#                 <div class="w-full px-3">
#                 <label class="block uppercase tracking-wide text-gray-100 text-xs font-bold mb-2" for="question">
#                     Ingrese su pregunta:
#                 </label>
#                 <input class="appearance-none block w-full bg-gray-200 text-gray-700 border border-gray-200 rounded py-3 px-4 mb-3 leading-tight focus:outline-none focus:bg-white" id="question" type="text" placeholder="¿Que deseas preguntar?">
#                 </div>
#             </div>
#             <div class="flex justify-center">
#                 <button class="bg-white hover:bg-gray-500 text-black font-bold py-2 px-4 rounded" type="button" onclick="sendQuestion()">
#                     Enviar pregunta
#                 </button>
#             </div>
#             </form>
#             <div class="w-full max-w-3xl p-8 rounded-lg mt-8">
#                 <div id="respuesta" class="text-gray-100"></div>
#                 <div id="loading" class="center">
#                     <div class="wave"></div>
#                     <div class="wave"></div>
#                     <div class="wave"></div>
#                     <div class="wave"></div>
#                     <div class="wave"></div>
#                     <div class="wave"></div>
#                     <div class="wave"></div>
#                     <div class="wave"></div>
#                     <div class="wave"></div>
#                     <div class="wave"></div>
#                 </div>
#             </div>
#             <div class="divide-y divide-dashed divide-gray-600 flex flex-col max-w-3xl p-4">
#                 <div></div>
#                 <div class="text-gray-600 ">
#                     <p class="my-4">
#                         Gracias por utilizar nuestra aplicación que donde conectamos con la API de OpenAI para proporcionar respuestas a tus preguntas sobre documentos especificos previamente cargado. 
#                     </p>
#                     <p class="my-4">
#                         Aunque hemos trabajado arduamente para asegurarnos de que las respuestas sean lo más precisas posible, <b>es importante</b> tener en cuenta que aun se puede mejorar y que a veces las respuestas pueden no ser completamente exactas. Aun hay algunas optimizaciones y pruebas que se pueden realizar para mejorar la calidad de respuesta en diferentes situaciones, a veces puede haber ambigüedad o complejidad en las preguntas que hacen que las respuestas no sean del todo precisas. Por eso, siempre recomendamos leer y evaluar críticamente las respuestas que obtiene, y no tomarlas como una verdad absoluta.
#                     </p>
#                     <p class="my-4">
#                         Además, ten en cuenta que la inteligencia artificial no es infalible y que puede haber errores en las respuestas que proporciona. El plan es trabajar para mejorar nuestra aplicación y la precisión de las respuestas que proporcionamos, pero agradecemos tus comentarios y sugerencias para ayudarnos a seguir mejorando.
#                     </p>
#                     <p class="my-4">
#                         Si deseas una presentación mas personalizada para tu modelo de negocio contactanos y conversamos.
#                     </p>
#                     <p class="my-4">
#                         ¡Gracias por utilizar nuestra aplicación y esperamos que encuentres las respuestas que necesitas!
#                     </p>

#                 </div>
#                 <div class="text-gray-600 ">
#                     <p class="my-4">
#                         Esto es demo creado por <b>Ponce.cloud</b> utilizando <b>OpenAI</b> aun no representa un proyecto 100% en producción.
#                     </p>
#                 </div>
#             </div>
#         </div>

#         <script>
#             document.getElementById("loading").hidden = true
    
#             function sendQuestion() {
#                 document.getElementById("loading").hidden = false
#                 document.getElementById("respuesta").innerHTML = ""

#                 var question = document.getElementById("question").value;
#                 var xhttp = new XMLHttpRequest();
#                 xhttp.onreadystatechange = function() {
#                     if (this.readyState == 4 && this.status == 200) {
#                         document.getElementById("loading").hidden = true
#                         document.getElementById("respuesta").innerHTML = this.responseText;
#                     }
#                 };
#                 xhttp.open("GET", "/answer?question=" + question, true);
#                 xhttp.send(); 
#             }
#         </script>
#     </body>
#     </html>
#     """
#     return form_html


# if __name__ == '__main__':
#     app.run(debug=True, port=os.getenv("PORT", default=5000))



# Regex pattern to match a URL
# Patrón Regex para coincidir con una URL
# HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

# Define root domain to crawl
# Definir dominio raíz para rastrear
domain = "leyes"



################################################################################
### Step 5
################################################################################

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ', regex=True)
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie


################################################################################
### Step 6
################################################################################

# Create a list to store the text files
# Crear una lista para almacenar los archivos de texto
# texts=[]

# # Get all the text files in the text directory
# # Obtenga todos los archivos de texto en el directorio de texto
# # for file in os.listdir("text/" + domain + "/"):
# for file in os.listdir("text/" + domain + "/"):

#     # Open the file and read the text
#     with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
#         text = f.read()

#         # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
#         texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

# # Create a dataframe from the list of texts
# # Crear un marco de datos de la lista de textos
# df = pd.DataFrame(texts, columns = ['fname', 'text'])

# # Set the text column to be the raw text with the newlines removed
# # Establecer la columna de texto para que sea el texto sin procesar con las líneas nuevas eliminadas
# df['text'] = df.fname + ". " + remove_newlines(df.text)
# df.to_csv('processed/scraped.csv')
# df.head()

################################################################################
### Step 7
################################################################################

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
# Cargue el tokenizador cl100k_base que está diseñado para funcionar con el modelo ada-002
# tokenizer = tiktoken.get_encoding("cl100k_base")

# df = pd.read_csv('processed/scraped.csv', index_col=0)
# df.columns = ['title', 'text']

# # Tokenize the text and save the number of tokens to a new column
# df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# # Visualize the distribution of the number of tokens per row using a histogram
# # Tokenize el texto y guarde el número de tokens en una nueva columna
# df.n_tokens.hist()

################################################################################
### Step 8
################################################################################

max_tokens = 500

# # Function to split the text into chunks of a maximum number of tokens
# # Función para dividir el texto en fragmentos de un número máximo de tokens
# def split_into_many(text, max_tokens = max_tokens):

#     # Split the text into sentences
#     sentences = text.split('. ')

#     # Get the number of tokens for each sentence
#     n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
#     chunks = []
#     tokens_so_far = 0
#     chunk = []

#     # Loop through the sentences and tokens joined together in a tuple
#     for sentence, token in zip(sentences, n_tokens):

#         # If the number of tokens so far plus the number of tokens in the current sentence is greater 
#         # than the max number of tokens, then add the chunk to the list of chunks and reset
#         # the chunk and tokens so far
#         if tokens_so_far + token > max_tokens:
#             chunks.append(". ".join(chunk) + ".")
#             chunk = []
#             tokens_so_far = 0

#         # If the number of tokens in the current sentence is greater than the max number of 
#         # tokens, go to the next sentence
#         if token > max_tokens:
#             continue

#         # Otherwise, add the sentence to the chunk and add the number of tokens to the total
#         chunk.append(sentence)
#         tokens_so_far += token + 1
        
#     # Add the last chunk to the list of chunks
#     if chunk:
#         chunks.append(". ".join(chunk) + ".")

#     return chunks
    

# shortened = []

# Loop through the dataframe
# # Bucle a través del marco de datos
# for row in df.iterrows():

#     # If the text is None, go to the next row
#     if row[1]['text'] is None:
#         continue

#     # If the number of tokens is greater than the max number of tokens, split the text into chunks
#     if row[1]['n_tokens'] > max_tokens:
#         shortened += split_into_many(row[1]['text'])
    
#     # Otherwise, add the text to the list of shortened texts
#     else:
#         shortened.append( row[1]['text'] )

################################################################################
### Step 9
################################################################################

# df = pd.DataFrame(shortened, columns = ['text'])
# df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
# df.n_tokens.hist()

################################################################################
### Step 10
################################################################################

# Note that you may run into rate limit issues depending on how many files you try to embed
# Please check out our rate limit guide to learn more on how to handle this: https://platform.openai.com/docs/guides/rate-limits
# Tenga en cuenta que puede encontrarse con problemas de límite de velocidad según la cantidad de archivos que intente incrustar
# Consulte nuestra guía de límite de tasa para obtener más información sobre cómo manejar esto: https://platform.openai.com/docs/guides/rate-limits

# df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
# df.to_csv('processed/embeddings.csv')
# df.head()

################################################################################
### Step 11
################################################################################

df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
df.head()

################################################################################
### Step 12
################################################################################


# Cree un contexto para una pregunta encontrando el contexto más similar del marco de datos
def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    # Obtener las incrustaciones para la pregunta.
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    # Obtener las distancias de las incrustaciones
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    # Ordene por distancia y agregue el texto al contexto hasta que el contexto sea demasiado largo
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        # Agrega la longitud del texto a la longitud actual
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        # Si el contexto es demasiado largo, romper
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        # De lo contrario, agréguelo al texto que se devuelve
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the questin and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"Lo siento mis respuestas esta limitadas a mi base de conocimiento sobre la LEY 2208 DEL 17 DE MAYO DE 2022 , disculpame si no puedo responderte o comprendo tu pregunta aun soy un modelos de aprendizaje en entrenamiento.\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:", 
            temperature=0.8,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

################################################################################
### Step 13
################################################################################

print(answer_question(df, question="Cuantos articulos tienes?", debug=False)) 


# @app.route("/answer")
# def answer():
#     question = request.args.get("question")
#     respuesta = answer_question(df, question=question, debug=False) 
#     return respuesta
