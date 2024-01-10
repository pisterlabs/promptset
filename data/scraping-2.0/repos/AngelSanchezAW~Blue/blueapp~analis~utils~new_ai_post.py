from analis.models import ArticuloGenerado
from openai import OpenAI
from consts import APIKEYOAI

client = OpenAI(api_key=APIKEYOAI)
from consts import APIKEYOAI

def new_ai_post(nombreSitioWeb, urlSitioWeb, postUrl, titulo_new_post, extracto_texto_new_post):

    # Establecer la clave de API de OpenAI


    prompt = f"Crea un articulo original optimizado para SEO con esta información: {titulo_new_post} {extracto_texto_new_post}"

    # Configurar el modelo de lenguaje
    modelo = "gpt-3.5-turbo"
    mensaje = [
        {"role":"system","content":"Eres un experto en redacción de articulos."},
        {"role":"user","content":prompt}
    ] 

    # Generar la respuesta utilizando la API de OpenAI
    response = client.chat.completions.create(model=modelo,
    messages=mensaje,
    temperature=1,
    max_tokens=2000)

    respuesta = response.choices[0].message.content

    # Crear una instancia de ArticuloGenerado y guardarla en la base de datos
    ai_post_instance = ArticuloGenerado.objects.create(
        contenido_generado=respuesta,
        titulo=titulo_new_post,
        nombre_sitio_web=nombreSitioWeb,
        url_sitio_web=urlSitioWeb,
        post_url=postUrl
    )

    print("Artículo generado con éxito y guardado en la base de datos")

    return ai_post_instance 
