import os
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key = os.environ["OPENAI_API_KEY"]

clinical_guiudes = ["C:/Users/pedro/OneDrive/Escritorio/Proyecto_IA_SS/Flask_API_ChatBot/gpt_based_texts/tamizaje.txt",
					"C:/Users/pedro/OneDrive/Escritorio/Proyecto_IA_SS/Flask_API_ChatBot/gpt_based_texts/diagnostico.txt",
					"C:/Users/pedro/OneDrive/Escritorio/Proyecto_IA_SS/Flask_API_ChatBot/gpt_based_texts/tratamiento.txt"]
context = []

for file in clinical_guiudes:
	with open(file, encoding="utf-8") as f:
		text = f.read()
		context.append(text)

# Tow different functions to get the completion from gpt api, the second is more robust

def get_completion(prompt, model="gpt-4"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    return response.choices[0].message["content"]

# Defining system behavior, basically prompt engineering

delimitador = "####"

system_message = f"""

Seguir los siguientes pasos para responder las búsquedas del usuario. 
Las búsquedas del usuario serán respecto al enfoque de alguna o todas de \
las siguientes categorías: Tamizaje, Diagnóstico y Tratamiento.
Estas categorías sirven como guía clínica para el usuario al momento de dar \
seguimiento a pacientes, en este caso particular a pacientes con diabetes mellitus.
Las búsquedas del usuario serán delimitadas con cuatro hashtags, i.e. {delimitador}.

Paso 1: {delimitador} Primero decidir qué tipo de búsqueda está realizando el usuario \
estas búsquedas pueden ser respecto a Tamizaje, Diagnóstico o Tratamiento \
por lo tanto solo existen tres categorías para clasificar la búsqueda del usuario.

Paso 2: {delimitador} Si la búsqueda es respecto a Tamizaje, se facilitaría para el usuario \
recibir información sobre recomendaciones y puntos positivos que son útiles \
como herramientas de tamizaje.

Escriba una síntesis de recomendaciones y utilidades basada en la información \
provista en la guía de Tamizaje delimitada por triples tildes.

Guía de Tamizaje: ```{context[0]}```

Paso 3: {delimitador} Si la búsqueda es sobre diagnóstico, el usuario espera conocer los \
síntomas de la enfermedad, recomendaciones para el diágnostico y críterios \
más específicos para el diagnóstico.

Escriba una sístesis de los síntomas de la enfermedad. \
Si se piden recomendaciones escriba una síntesis de estas recomendaciones \
y resalte aspectos importantes.
Si se piden criterios de diagnóstico haga una lista de estos criterios.
Para cualquiera de las búsquedas de usuario sobre síntomas, recomendaciones o \
criterios de diagnóstico, utilizar la guía de diagnóstico delimitada por triples tildes.

Guía de Tamizaje: ```{context[1]}```

Paso 4: {delimitador} Si la búsqueda es sobre tratamiento, el usuario \
espera conocer las recomendaciones y no recomendaciones de tratamiento \
asi como también los fármacos quese pueden utilizar.

Si el usuario solo requiere recomendaciones genere una síntesis de las recomendaciones \
de tratamiento y considere también lo que no es recomendado.
Si el usuario pide conocer los fármacos para el tratamiento de la enfermedad, haga una lista de ellos \
y resalte sus características en caso de que tengan.

Para lo anterior utilice la guía de tratamiento que se muestra a continuación delimitada por triples tildes.

Guía de Tratamiento: ```{context[2]}```

Paso 5: {delimitador} Si el usuario realiza una búsqueda sobre algo diferente a Tamizaje, Diagnóstico o Tratamiento de 
la enfermedad, considere de una manera amable indicarle al usuario que sus funciones son enfocadas \
en alguna de las tres categorías mencionadas anteriormente. Responda al usuario de amablemente.

Utilice el siguiente formato:
Paso 1:{delimitador} <razonamiento del paso 1>
Paso 2:{delimitador} <razonamiento del paso 2>
Paso 3:{delimitador} <razonamiento del paso 3>
Paso 4:{delimitador} <razonamiento del paso 4>
Respuesta al usuario:{delimitador} <respuesta al cliente>

Asegúrese de incluir {delimitador} para separar cada paso.
"""


user_message = f"""

¿qué hago si mi paciente es diagnosticado con diabetes?

"""
messages =  [  
{'role':'system', 
 'content':system_message},    
{'role':'user', 
 'content': f"{delimitador}{user_message}{delimitador}"},  
] 



if __name__ == "__main__":

	response = get_completion_from_messages(messages)
	
	try:
		final_response = response.split(delimitador)[-1].strip()
	except Exception as e:
		final_response = "Lo siento, tengo problemas en este momento, intente hacer otra pregunta."

	print(final_response)