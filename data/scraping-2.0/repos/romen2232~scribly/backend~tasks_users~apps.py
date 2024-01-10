from django.apps import AppConfig
import openai   
import requests 
import os
import re
class TasksUsersConfig(AppConfig):
    """Configuration class for the 'tasks_users' Django application."""
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'tasks_users'

def evaluation(input, statement):
    """
    Evaluate a student's text response based on a provided statement using OpenAI's GPT-3.5 model.

    Parameters:
    - input (str): The student's text response to be evaluated.
    - statement (str): The statement of the activity.

    Returns:
    - tuple: Contains the evaluated score (float) and the detailed response (str) from GPT-3.5.
    """
    
    # Set OpenAI API key
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Set the API endpoint

    URL = "https://api.openai.com/v1/chat/completions"
    
    # statement = "Escribe una historia sen la que el protagonista sea un robot del fututo"
    difficulty=3
# Descriptions for the difficulty levels    
    difficulty_descriptions = [
        "Sé amable y enfócate en los aspectos positivos, incluso si hay áreas de mejora.",
        "Proporciona una crítica equilibrada, destacando tanto los aspectos positivos como las áreas de mejora.",
        "Sé objetivo y evalúa el texto de manera justa en todos los aspectos.",
        "Sé crítico y no dudes en señalar áreas de mejora, manteniendo un tono respetuoso.",
        "Sé muy crítico y riguroso en tu evaluación, buscando áreas de mejora en todos los aspectos del texto."
    ]
    critical_level = difficulty_descriptions[difficulty - 1]
    
    system_text= f"Eres un asistente de enseñanza amigable y servicial. Explicas conceptos de gran profundidad en pocas palabras,  usando términos simples y das ejemplos para ayudar a las personas a aprender. Proporcionas respuestas personalizadas que ayudan  al usuario a mejorar sus puntos débiles y darse cuenta de sus puntos fuertes.\n\nTu enfoque es {critical_level}."
    user_text=f'''
    Por favor, evalúa el siguiente texto para saber si cumple ESTRICTAMENTE el enunciado de la actividad, siguiendo los criterios detallados a continuación y presenta los resultados en un formato específico:

**Información de referencia:**
- **ENUNCIADO de la actividad:** {statement}
- **TEXTO propuesto por el alumno:** {input}

**Instrucciones de Evaluación:**

1. **Escala de Evaluación:**
   - **0 puntos:** Si el texto no se alinea EXACTAMENTE con el enunciado. Cualquier desviación del enunciado, por mínima que sea, debe resultar en una puntuación de 0 en esta categoría.
   - **1 a 4 puntos:** Si el texto tiene relación con el enunciado, pero es demasiado breve o simple.
   - **5 a 10 puntos:** Si el texto cumple adecuadamente con el enunciado y muestra un nivel de elaboración.

2. **Atributos a evaluar:**
   - **Cumplimiento del enunciado:** Debes ser extremadamente estricto en este aspecto. Si el texto no se alinea exactamente con el enunciado, la puntuación será 0. No hay excepciones.
   - **Calidad literaria:** Evalúa la gramática, la puntuación y la estructura del texto.
   - **Creatividad y originalidad:** Considera la innovación y la singularidad del texto.
   - **Coherencia:** Asegúrate de que el texto tenga un flujo lógico y coherente.

3. **Proceso de Reflexión:** Antes de tomar una decisión sobre la puntuación, tómate un momento para reflexionar sobre el texto y considerar todos los aspectos mencionados.

**Formato de Presentación de la Evaluación:**

- **Puntuación:** n/10
- **Cumplimiento del enunciado:** [Análisis detallado]
- **Calidad literaria:** [Análisis detallado]
- **Creatividad y originalidad:** [Análisis detallado]
- **Coherencia:** [Análisis detallado]
- **Resumen:** [Resumen general de los análisis y la puntuación otorgada]**Formato de Presentación de la Evaluación:**

**Formato de Rechazo por no cumplir el encunciado:**

- **Puntuación:** 0/10
- **Cumplimiento del enunciado:** [Justificacion razonada, satirica e ingeniosa]


    '''

    
    

    payload = {
        "model": "gpt-3.5-turbo",
        "temperature": 1.0,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    # Make the API request
    laresponse = requests.post(URL, headers=headers, json=payload).json()
    laresponse = laresponse['choices'][0]['message']['content']

    # Extract the final score from the message
    note = re.search(r'(\d+(\.\d+)?)/10', laresponse)
    if note:
        note = float(note.group(1))
    else:
        note = -10 # Handle this case as preferred

    return note, laresponse

def CorrectionWrite(input, statement):
    """
    Determine if a student's text response passes the evaluation criteria.

    Parameters:
    - input (str): The student's text response.
    - statement (str): The statement of the activity.

    Returns:
    - tuple: Contains the correction status (bool), detailed response (str) and evaluated score (float).
    """
    
    mark, response = evaluation(input=input, statement=statement)

    if mark > 4:
        correction = True
    else:
        correction = False

    return correction, response, mark