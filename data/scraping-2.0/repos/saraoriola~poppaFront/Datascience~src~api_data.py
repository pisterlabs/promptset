from spanlp.palabrota import Palabrota
from spanlp.domain.strategies import Preprocessing, RemoveUserMentions, RemoveUrls, RemoveHashtags, RemoveEmoticons, RemoveEmailAddress
from flask import Flask, jsonify, request
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import timedelta
import pandas as pd
import psycopg2
import requests
import openai
import deepl
import os
from dotenv import load_dotenv
import locale

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Asignar variables de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROMPT_FEEDBACK = os.getenv('OPENAI_PROMPT_FEEDBACK')
openai.api_key = OPENAI_API_KEY

# Asignar variables de DeepL
DEEPL_API_KEY = os.getenv('DEEPL_API_KEY')
translator = deepl.Translator(DEEPL_API_KEY)

# Asignar variables de AWS
AWS_USER = os.getenv('AWS_USER')
AWS_PASS = os.getenv('AWS_PASS')
AWS_HOST = os.getenv('AWS_HOST')
AWS_PORT = os.getenv('AWS_PORT')
AWS_NAME = os.getenv('AWS_NAME')

# Asignar variables de RENDER
RENDER_USER = os.getenv('RENDER_USER')
RENDER_PASS = os.getenv('RENDER_PASS')
RENDER_HOST = os.getenv('RENDER_HOST')
RENDER_PORT = os.getenv('RENDER_PORT')
RENDER_NAME = os.getenv('RENDER_NAME')


# Instanciar funciones de bad languaje
palabrota = Palabrota()
strategies = [RemoveEmailAddress(), RemoveUrls(), RemoveUserMentions(), RemoveHashtags(), RemoveEmoticons()]

# Función que procesa la api de OpenAI para el feedback
def api_openai_feedback(inputs):
    """
    Esta función utiliza el modelo GPT-3.5 Turbo de OpenAI para proporcionar una respuesta basada en un mensaje de entrada.
    
    Args:
    inputs (str): El mensaje de entrada que se enviará al modelo.

    Returns:
    dict: Un diccionario que contiene la respuesta generada por el modelo, incluyendo información como el texto de la respuesta, la probabilidad, etc.
    """
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": OPENAI_PROMPT_FEEDBACK
        },
        {
        "role": "user",
        "content": str(inputs)
        }
    ],
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response

# Función que procesa la api de OpenAI para generar una imagen de portada
def api_openai_cover_image(inputs, size="1024x1024"): 
    """
    Genera una solicitud a la API de generación de imágenes de OpenAI para crear una imagen realista basada en el texto de entrada proporcionado.
    
    Parámetros:
    inputs (str): Un texto descriptivo de entrada que proporciona instrucciones para el proceso de generación de imágenes.
    size (str, opcional): El tamaño deseado de la imagen generada en el formato "anchoxalto" (por ejemplo, "1024x1024").
                         El valor predeterminado es "1024x1024".
    
    Retorna:
    str: URL de la imagen generada.
    str: El tamaño de la imagen generada.
    """
    response = openai.Image.create(
    prompt=f'Create realistic image of {inputs} with a cartoon details and without landscape. The color palette should be composed of soothing neutral and cream tones, and the background should be a pure, crisp white.',
    n=1,
    size=size
    )
    return response['data'][0]['url'], size

# URL de la API de DeepL para obtener los idiomas disponibles
def api_deepl_languages(language='English (British)'):
    """
    Recupera los idiomas compatibles con DeepL y sus códigos, y devuelve el código para un idioma especificado.

    Parámetros:
    language (str, opcional): El nombre del idioma para el cual deseas recuperar el código.
                             El valor predeterminado es 'Inglés (Británico)'.

    Retorna:
    str: El código de idioma para el idioma especificado.
    
    Nota:
    - Debes tener una clave API válida de DeepL almacenada en la variable DEEPL_API_KEY para usar esta función.
    - Asegúrate de configurar un agente de usuario en los encabezados para cumplir con las pautas de uso de la API.
    """
    url = "https://api-free.deepl.com/v2/languages?type=target"
    headers = {
        "Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}",
        "User-Agent": "YourApp/1.2.3"
    }
    response = requests.get(url, headers=headers)
    dict_languages = {}
    for item in response.json():
        dict_languages[item['name']] = item['language']
    return dict_languages[language]



app = Flask(__name__)



# HOME
@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello, World!'



# API PARA BAD LANGUAGE Y SENTIMENT ANALYSIS
@app.route('/get_sentiment_analysis_feedback', methods=['POST'])
def sentiment_analysis_model():
    """
    Analiza el sentimiento y realiza diversas operaciones en el feedback proporcionado.

    Esta API recibe feedback a través del cuerpo de una solicitud POST y realiza las siguientes tareas:

    1. Verifica si el feedback contiene palabrotas.
    2. Limpia el texto del feedback según estrategias definidas antes de enviarlo a la API de OpenAI.
    3. Realiza un análisis de sentimiento sobre el feedback utilizando la API de OpenAI.
    4. Procesa y presenta los resultados en un formato legible.

    Parameters:
    Ninguno explícito (los datos de entrada se toman del cuerpo de la solicitud POST).

    Returns:
    jsonify(dict): Un objeto JSON que contiene información sobre el feedback, incluyendo si contiene palabrotas,
                   los resultados del análisis de sentimiento y detalles sobre el uso de la API de OpenAI.

    Nota:
    - Esta API está diseñada para ser utilizada con una solicitud POST que proporcione un objeto JSON con un campo
      llamado 'feedback' que contenga el texto del feedback a analizar.
    - Se asume que la función 'api_openai_feedback' se encuentra definida en otra parte del código y maneja la
      comunicación con la API de OpenAI.
    - Los resultados se presentan en un formato JSON que puede incluir información sobre el contenido del feedback,
      resultados de análisis de sentimiento y detalles de uso de la API de OpenAI.
    - Si se produce un error durante el proceso, se devolverá una respuesta JSON con un mensaje de error y un código
      de estado 400 (Bad Request).
    """
    # Recoge el feedback del Body
    data = request.get_json()
    feedback = data['feedback'].encode('utf-8').decode('utf-8')
    dict_data = {}
    
    # Comprueba si tiene una palabrota
    swearword = palabrota.contains_palabrota(feedback)
    dict_data['swearword'] = swearword

    # Limpia el texto del feedback antes de pasarlo por la API de OpenAI
    preprocessor = Preprocessing(data=feedback, clean_strategies=strategies)
    clean_feedback = preprocessor.clean().split('.')

    # Análisis de sentimiento sobre el feedback con la API de OpenAI
    valuation = api_openai_feedback(clean_feedback)

    try:
        # Limpiar el formato del output de la función de OpenAI
        data = valuation['choices'][0]['message']['content']
        data = data.replace('{', '').replace('}', '').replace(',', '').replace('"', '').strip().split('\n')
        for item in data:
            prov = item.strip().split(':')
            key, value = prov[0], prov[1]
            dict_data[key] = int(value)

        # Contabilidaz del coste
        usage = valuation['usage']
        prov_dict = {}
        for key, value in usage.items():
            prov_dict[key] = int(value)
            dict_data['usage'] = prov_dict
            
    except:
        response = {"ERROR": "Parece ser un error en el proceso de limpieza de los datos de la API utilizada para el feedback."}
        return jsonify(response), 400
    
    return jsonify(dict_data), 200



# API PARA IMAGE MODEL
@app.route('/get_cover_image', methods=['POST'])
def cover_image():
    """
    Genera una imagen de portada basada en el texto proporcionado utilizando la API de OpenAI.

    Esta API toma un objeto JSON del cuerpo de una solicitud POST que contiene un campo llamado 'text'. El proceso
    consta de los siguientes pasos:

    1. Recoge el texto del campo 'text' en el objeto JSON de entrada.
    2. Traduce el texto a inglés británico (EN-GB) utilizando un servicio de traducción.
    3. Genera una imagen de portada utilizando la API de OpenAI basada en el texto traducido.
    4. Devuelve la URL de la imagen generada y el costo estimado de acuerdo con el tamaño de la imagen.

    Parameters:
    Ninguno explícito (los datos de entrada se toman del cuerpo de la solicitud POST).

    Returns:
    jsonify(dict): Un objeto JSON que contiene la URL de la imagen generada y el costo estimado.
    
    Nota:
    - Esta API espera recibir una solicitud POST con un objeto JSON que tenga un campo llamado 'text' que contenga el texto
      que se utilizará para generar la imagen.
    - El texto se traduce a inglés británico (EN-GB) antes de enviarlo a la API de OpenAI.
    - El costo estimado se calcula en función del tamaño de la imagen generada y se devuelve en el objeto JSON de salida.
    """
    # Recoge el feedback del Body
    data = request.get_json()
    text = data['text'].encode('utf-8').decode('utf-8')

    # Traducir texto_body a inglés
    translation = translator.translate_text(text, target_lang="EN-GB")

    # Obtener url imagen generada con OpenAI
    ouput_api = api_openai_cover_image(translation.text)
    url_image = ouput_api[0]
    size_image = ouput_api[1]
    
    # Diccionario que ajusta el coste que supone generar una imagen con OpenAI
    dict_size = {
        "1024x1024": 0.020,
        '512×512': 0.018,
        '256×256': 0.016,
    }
    
    # Crear diccionario de salida con la URL y el coste
    dict_url = {
        "url": url_image,
        "usage": dict_size[size_image]
    }

    return jsonify(dict_url), 200



# API PARA TRANSLATION MODEL
@app.route('/get_realtime_translation', methods=['POST'])
def realtime_translation():
    """
    Realiza traducción en tiempo real de texto a un idioma especificado utilizando un servicio de traducción.

    Esta API toma un objeto JSON del cuerpo de una solicitud POST que debe contener los siguientes campos:
    - 'text': El texto que se va a traducir.
    - 'language': El idioma al que se debe traducir el texto.

    Los pasos que realiza la API son los siguientes:

    1. Recoge el texto y el idioma del objeto JSON de entrada.
    2. Verifica si se especificó un idioma para la traducción.
    3. Consulta la API para obtener el código de idioma correspondiente al idioma especificado.
    4. Utiliza un servicio de traducción para traducir el texto al idioma especificado.
    5. Devuelve un objeto JSON que contiene el texto original y el texto traducido.

    Parameters:
    Ninguno explícito (los datos de entrada se toman del cuerpo de la solicitud POST).

    Returns:
    jsonify(dict): Un objeto JSON que contiene el texto original y el texto traducido.
    
    Nota:
    - Esta API espera recibir una solicitud POST con un objeto JSON que contenga los campos 'text' y 'language' para
      realizar la traducción.
    - La API consulta un servicio para obtener el código de idioma correspondiente antes de realizar la traducción.
    - El texto traducido se devuelve en el objeto JSON de salida junto con el texto original.
    - Si no se especifica un idioma en la solicitud, la API no realizará ninguna traducción y no devolverá ningún resultado.
    """
    # Recoge el feedback del Body
    data = request.get_json()
    text = data['text'].encode('utf-8').decode('utf-8')
    language = data['language'].encode('utf-8').decode('utf-8')

    if language == None:
        return 

    # Recibir item del idioma
    item_language = api_deepl_languages(language)

    # Traducir texto_body a un idioma
    if 'EN' in item_language:
        translation = translator.translate_text(text, target_lang=item_language)
    else:
        translation =translator.translate_text(text, target_lang=item_language, formality="more")

    # Diccionario que muestra texto original y texto traducido
    dic_languages = {
        'original': text,
        'translation': str(translation)
    }

    return jsonify(dic_languages), 200



# API DASHBOARD
@app.route('/get_dashboard', methods=['POST'])
def dashboard():
    """
    Obtiene información detallada de un evento a partir de su título.

    Esta API toma el título de un evento como entrada y recopila información detallada sobre ese evento desde una base de datos. La información incluye detalles del evento, como fecha y hora, orador, descripción, capacidad de ubicación, así como estadísticas sobre los asistentes registrados y confirmados, nacionalidades de los asistentes, tipos de organizaciones presentes y un registro de entradas y salidas.

    Parameters:
        event_title (str): El título del evento del cual se desea obtener información.

    Returns:
        jsonify(dict): Un objeto JSON que contiene datos detallados sobre el evento, incluyendo detalles del evento en sí, estadísticas de asistentes, nacionalidades, tipos de organizaciones y un registro de entradas y salidas.

    Nota:
    - Esta API espera recibir una solicitud POST con un objeto JSON que contenga un campo llamado 'event_title' que especifique el título del evento a consultar.
    - La información se recopila de una base de datos utilizando SQLAlchemy y se presenta en un formato JSON estructurado para su fácil lectura y procesamiento.
    - Los datos proporcionados incluyen detalles generales del evento, estadísticas sobre los asistentes y datos específicos sobre la entrada y salida de asistentes durante el evento.
    """


    data = request.get_json()
    event_title = data['event_title']

    # # Conexión con la BD de AWS
    # db_url = f'mysql+mysqlconnector://{AWS_USER}:{AWS_PASS}@{AWS_HOST}:{AWS_PORT}/{AWS_NAME}'
    # engine = create_engine(db_url)

    # Conexión con la BD de RENDER
    db_url = f'postgresql+psycopg2://{RENDER_USER}:{RENDER_PASS}@{RENDER_HOST}:{RENDER_PORT}/{RENDER_NAME}'
    engine = create_engine(db_url)


    query = f"""
    SELECT
        "E"."dateTime",
        "E"."speacker",
        "E"."description",
        "E"."title",
        "E"."location_id",
        "L"."capacity",
        "U"."country",
        "O"."name",
        "T"."name",
        "EU"."arriveTime",
        "EU"."leaveTime",
        COUNT(DISTINCT "T"."name"),
        COUNT(DISTINCT "U"."country"),
        "U"."confirmed"
    FROM
        "Events" AS "E"
    JOIN
        "Locations" AS "L" ON "E"."location_id" = "L"."id"
    LEFT JOIN
        "EventUsers" AS "EU" ON "E"."id" = "EU"."event_id"
    LEFT JOIN
        "Users" AS "U" ON "EU"."user_id" = "U"."id"
    LEFT JOIN
        "Organizations" AS "O" ON "U"."organization_id" = "O"."id"
    LEFT JOIN
        "Types" AS "T" ON "O"."type_id" = "T"."id"
    WHERE
        "E"."title" = '{event_title}'
    GROUP BY
        "E"."dateTime",
        "E"."speacker",
        "E"."description",
        "E"."title",
        "E"."location_id",
        "L"."capacity",
        "U"."country",
        "O"."name",
        "T"."name",
        "EU"."arriveTime",
        "EU"."leaveTime",
        "U"."confirmed";
    """



    # Crea una sesión de SQLAlchemy
    Session = sessionmaker(bind=engine)
    session = Session()

    with session:
        # Ejecuta la consulta para leer la tabla Facilities
        database = session.execute(text(query))
        columnas = database.keys()

        # Recupera los resultados
        facilities_data = database.fetchall()

        # bucle for que corre todos los usuarios
        dict_table = {}
        horas_entrada = []
        horas_salida = []
        entry_exit_Entradas = {}
        entry_exit_Salidas = {}
        nationality = {}
        types = {}
        registered = 0
        confirmed = 0
        present = 0

    for row in facilities_data:
        locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
        dict_table['title'] = row[3]
        dict_table['description'] = row[2]
        dict_table['speaker'] = row[1]
        dict_table['day'] = str(pd.to_datetime(row[0], yearfirst=True).strftime('%A')).capitalize()
        dict_table['month'] = str(pd.to_datetime(row[0], yearfirst=True).strftime('%B')).capitalize()
        registered += int(row[11])
        confirmed += int(row[13])
        present += [1 if row[9] != None else 0][0]
        dict_table['capacity'] = row[5]

        # ENTRY_EXIT
        horas_entrada.append(row[9])
        horas_salida.append(row[10])
        
        horas_evento = []

        if row[9] == None:
            horas_evento.append(None)
        else:
            if row[10] == None:
                horas_evento.append(None)
            else:
                fecha_inicio = pd.to_datetime(row[9])
                fecha_fin = pd.to_datetime(row[10])
                hora_abierto = (fecha_inicio - timedelta(hours=1, minutes=5)).to_pydatetime()
                hora_cierre = (fecha_fin + timedelta(minutes=30)).to_pydatetime()
                while hora_abierto < hora_cierre:
                    minutos_redondeados = (hora_abierto.minute // 5) * 5
                    timestamp_redondeado = hora_abierto.replace(minute=minutos_redondeados, second=0)
                    horas_evento.append(pd.to_datetime(timestamp_redondeado))
                    hora_abierto += timedelta(minutes=5)

        for fecha in horas_evento:
            if fecha in horas_entrada:
                count = horas_entrada.count(fecha)
                if fecha not in entry_exit_Entradas.keys():
                    entry_exit_Entradas[f'{fecha}'] = count
                else:
                    entry_exit_Entradas[f'{fecha}'] = count

            if fecha in horas_salida:
                count = horas_salida.count(fecha)
                if fecha not in entry_exit_Salidas.keys():
                    entry_exit_Salidas[f'{fecha}'] = count
                else:
                    entry_exit_Salidas[f'{fecha}'] = count

        # NACIONALIDAD
        if row[6] not in nationality.keys():
            nationality[row[6]] = row[12]
        else:
            nationality[row[6]] += row[12]

        # TYPE
        if row[8] not in types.keys():
            types[row[8]] = {
                row[7]: row[11]
            }
        elif row[7] not in types[row[8]].keys():
            types[row[8]][row[7]] = row[11]
        else:
            types[row[8]][row[7]] += row[11]


    dict_table['attendees'] = {'registered': registered, 
                            'confirmed': confirmed, 
                            'present': present}

    list_types = []
    for key, value in types.items():
        for company, count in value.items():
            list_types.append({'id':key, 'typeName':company, 'typeCount':count})

    dict_table['type'] = list_types

    dict_table['nationality'] = [{'country': country, 'userCount': count} for country, count in nationality.items()]

    dict_table['entry_exit'] = [{'id': 'Entradas', 'data': [{'x': pd.to_datetime(fecha), 'y': count} if fecha != 'None' else {'x': fecha, 'y': count} for fecha, count in entry_exit_Entradas.items()]}, 
                                {'id': 'Salidas', 'data': [{'x': pd.to_datetime(fecha), 'y': count} if fecha != 'None' else {'x': fecha, 'y': count} for fecha, count in entry_exit_Salidas.items()]}]


    return jsonify(dict_table)
    
            

if __name__ == '__main__':
    app.run(debug=True)
