#!/home/emi/Escritorio/Entorno/env/bin/python
import os
import json
import openai
import argparse
from dotenv import load_dotenv, find_dotenv

if not load_dotenv(find_dotenv()):
    raise RuntimeError('No se encontro el archivo .env o no se pudo cargar')

cliente = openai.OpenAI()

idiomas={   "es":"español",
            "en":"ingles",
            "fr":"frances",
            "it":"italiano",
            "pt":"portugues",
            "ca":"catalan",
            "de":"aleman",
            "ru":"ruso",
            "ja":"japones",
            "zh":"chino",
            "ar":"arabe",
            "pl":"polaco",
            "nl":"holandes",
            "el":"griego",
            "he":"hebreo",
            "hi":"hindi",
         }
        
cliente = openai.OpenAI()

def traducir_texto_markdown(texto, idioma_destino='es'):
    """
    Traduce el 'texto' dado al idioma especificado en 'idioma_destino'.

    Parámetros:
    - texto (str): El texto que se desea traducir.
    - idioma_destino (str): El idioma objetivo para la traducción. El valor predeterminado es 'es' (español).

    Retorna:
    - str: El texto traducido en el idioma especificado.
    """
    idioma = idiomas.get(idioma_destino)
    respuesta_api = cliente.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": f"Sos un Experto en formato markdown\
            y tu tarea es traducir celdas de jupyter notebook\
            manteniendo el formato de la celda sin agregar nada mas.\
            si hay formulas en latex, traducir literalmente solo el texto\
            si hay etiquetas html, mantenelas y solo traduce el texto"
        },
            {
                "role": "user",
                "content": f"Traducir literalmente (SIN AGREGAR NADA MAS) al {idioma} el siguente texto:{texto}"
            }
        ],
        max_tokens=2048,
        temperature = 0
    )
    return respuesta_api.choices[0].message.content


def traducir_archivo(archivo, idioma_destino='es'):
    """
    Traduce el archivo especificado al idioma de destino.

    Parámetros:
    - archivo (str): La ruta del archivo a traducir.
    - idioma_destino (str): El idioma objetivo para la traducción. Por defecto es 'es' (español).

    Retorna:
    - str: La ruta del archivo traducido.
    """
    
    nombre, extencion = os.path.splitext(archivo)
    if extencion != '.ipynb':
        raise ValueError("El archivo debe ser un notebook de jupyter")  
    with open(archivo, 'r') as f:
        notebook = json.load(f)
    for celda in notebook['cells'][:]:
        if celda['cell_type'] == 'markdown':
            texto = "".join(celda["source"])
            traduccion = traducir_texto_markdown(texto, idioma_destino)
            celda['source'] = [linea +"\n" 
                               if linea 
                                else "\n" 
                                for linea  in traduccion.split("\n")]
    destino = f"{nombre}_{idioma_destino}{extencion}"
    with open(destino, 'w') as f:
        json.dump(notebook, f)
    return destino

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archivo", type=str, help="El archivo a traducir")
    parser.add_argument("-e", type=str, help="El idioma al que se va a traducir el archivo", default="es")
    args = parser.parse_args()

    # Traduce el archivo al idioma especificado
    try:
        traducir_archivo(args.archivo, idioma_destino=args.e)
    except Exception as error:
        print("Error al traducir el archivo: {}".format(error))
    else:
        print("El {} se ha traducido al idioma {}.".format(args.archivo,args.e))

if __name__ == "__main__":
    main()
