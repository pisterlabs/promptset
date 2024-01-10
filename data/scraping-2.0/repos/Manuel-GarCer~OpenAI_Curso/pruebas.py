import openai

from IPython.display import display, HTML

openai.api_key =  "TU_API_KEY creada en https://platform.openai.com"


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


# Principios de incitación¶
# Principio 1: Escriba instrucciones claras y específicas
# Principio 2: Dar tiempo al modelo para "pensar"
# Tácticas
# Táctica 1: Utilizar delimitadores para indicar claramente las distintas partes de la entrada.
# Los delimitadores pueden ser ```, """, < >, <tag> </tag>, :

# text = f"""
# Debe expresar lo que quiere que haga un modelo proporcionando instrucciones tan claras como sea posible. 
# instrucciones tan claras y específicas 
# tan claras y específicas como sea posible. \ 
# Esto guiará al modelo hacia la salida deseada, y reducirá las posibilidades de recibir instrucciones irrelevantes. 
# y reducirá las posibilidades de recibir respuestas irrelevantes o incorrectas. 
# irrelevantes o incorrectas. No confunda escribir una 
# No confunda escribir un mensaje claro con escribir un mensaje corto. \ 
# En muchos casos, las instrucciones más largas proporcionan más claridad y contexto para el modelo, lo que puede dar lugar a respuestas más claras. 
# y contexto para el modelo, lo que puede conducir a 
# resultados más detallados y relevantes.
# """
# prompt = f"""
# Resume el texto delimitado por tres puntos \ 
# en una sola frase.
# ```{text}```
# """
# response = get_completion(prompt)
# print(response)

###################################################################################

# Táctica 2: Pedir una salida estructurada¶
# JSON, HTML

# prompt = f"""
# Genere una lista de tres títulos de libros inventados junto \ 
# con sus autores y géneros. 
# Proporciónelos en formato JSON con las siguientes claves: 
# book_id, title, author, genero
# """
# response = get_completion(prompt)
# print(response)

###################################################################################

# Táctica 3: Pedir al modelo que compruebe si se cumplen las condiciones

# text_1 = f"""
# Preparar una taza de té es muy fácil. En primer lugar, usted necesita para obtener algunos \ 
# agua hirviendo. Mientras eso sucede, \ 
# coge una taza y pon una bolsita de té en ella. Cuando el agua esté 
# lo suficientemente caliente, viértela sobre la bolsita de té. \ 
# Déjalo reposar un rato para que el té se infusione. Después de 
# minutos, saca la bolsita. Si quieres 
# Si quieres, puedes añadir un poco de azúcar o leche al gusto. \ 
# Y ¡listo! Ya tienes una deliciosa 
# taza de té para disfrutar.
# """
# prompt = f"""
# Se le proporcionará un texto delimitado por comillas triples. 
# Si contiene una secuencia de instrucciones, \ 
# reescriba esas instrucciones en el siguiente formato:

# Paso 1 - ...
# Paso 2 - ...
# ...
# Paso N - ...

# Si el texto no contiene una secuencia de instrucciones, \ 
# entonces simplemente escriba \ "No se proporcionan pasos.\"

# \"\"\"{text_1}"\"\"\"
# """
# response = get_completion (prompt)
# print("Finalización del texto 1:")
# print(response)

###################################################################################

# text_2 = f"""
# El sol brilla intensamente hoy, y los pájaros están \
# cantando. Es un hermoso día para ir a dar un \ 
# pasear por el parque. Las flores están floreciendo, y los árboles... 
# árboles se mecen suavemente con la brisa. La gente 
# está fuera, disfrutando del buen tiempo. \ 
# Algunos hacen picnic, otros juegan o simplemente se relajan en la hierba. 
# juegos o simplemente se relajan en la hierba. Es un día 
# día perfecto para pasar tiempo al aire libre y apreciar la \ 
# la belleza de la naturaleza.
# """
# prompt = f"""
# Se le proporcionará un texto delimitado por comillas triples. 
# Si contiene una secuencia de instrucciones, \ 
# reescriba esas instrucciones en el siguiente formato:

# Paso 1 - ...
# Paso 2 - ...
# ...
# Paso N - ...

# Si el texto no contiene una secuencia de instrucciones, \ 
# entonces simplemente escriba \ "No se proporcionan pasos.\"

# \"\"\"{text_2}"\"\"\"
# """
# response = get_completion (prompt)
# print("Finalización del texto 2:")
# print(response)

###################################################################################

# Táctica 4: "Pocos tiros" de aviso

# prompt = f"""
# Tu tarea es responder con un estilo coherente.

# <niño>: Enséñame sobre la paciencia.

# <abuelo>: El río que esculpe el más profundo \ 
# valle más profundo fluye de un modesto manantial. 
# más grandiosa sinfonía se origina en una sola nota; \ 
# el tapiz más intrincado comienza con un hilo solitario.

# <child>: Enséñame sobre la resiliencia.
# """
# response = get_completion (prompt)
# print(response)

###################################################################################

# Principio 2: Dar tiempo al modelo para "pensar"
# Táctica 1: Especificar los pasos necesarios para completar una tarea

# text = f"""
# En un encantador pueblo, los hermanos Jack y Jill emprenden \ 
# a buscar agua de un pozo en lo alto de una colina.
# Mientras subían, cantando alegremente, la desgracia los golpeó. 
# Jack tropezó con una piedra y cayó colina abajo. 
# y Jill le siguió. \ 
# Aunque algo maltrecha, la pareja regresó a casa... 
# abrazos reconfortantes. A pesar del percance. 
# sus espíritus aventureros permanecieron intactos, y continuaron 
# siguieron explorando con deleite.
# """
# # ejemplo 1
# prompt_1 = f"""
# Realice las siguientes acciones: 
# 1 - Resumir el siguiente texto delimitado por triple \
# con 1 frase.
# 2 - Traduzca el resumen al francés.
# 3 - Listar cada nombre en el resumen francés.
# 4 - Generar un objeto json que contenga lo siguiente
# claves: french_summary, num_names.

# Separe sus respuestas con saltos de línea.

# Texto:
# ```{text}```
# """
# response = get_completion (prompt_1)
# print("Respuesta a la pregunta 1:")
# print(response)

# prompt_2 = f"""
# Su tarea consiste en realizar las siguientes acciones 
# 1 - Resumir el siguiente texto delimitado por 
#   <> con 1 frase.
# 2 - Traduzca el resumen al francés.
# 3 - Listar cada nombre en el resumen francés.
# 4 - Dar salida a un objeto json que contenga las 
#   siguientes claves: french_summary, num_names.

# Utilice el siguiente formato:
# Texto: <texto a resumir>
# Resumen: <resumen>
# Traducción: <traducción del resumen>
# Nombres: <lista de nombres en italiano resumen>
# JSON de salida: <json con resumen y número_nombres>

# Texto: <{text}>
# """
# response = get_completion (prompt_2)
# print("respuesta para pregunta 2:")
# print(response)

###########################################################################################

# Táctica 2: Pedir al modelo que elabore su propia solución antes de llegar a una conclusión precipitada.

# prompt = f"""
# Determine si la solución del alumno es correcta o no.

# Pregunta:
# Estoy construyendo una instalación de energía solar y necesito
#  ayuda para calcular los costes. 
# - El terreno cuesta 100 $ / pie cuadrado
# - Puedo comprar paneles solares por 250 $ / pie cuadrado
# - He negociado un contrato de mantenimiento que me costará 
# 100 mil dólares al año, y 10 dólares adicionales por pie cuadrado.
# pie cuadrado
# ¿Cuál es el coste total para el primer año de operaciones 
# en función del número de pies cuadrados.

# Solución del alumno:
# Sea x el tamaño de la instalación en pies cuadrados.
# Costes:
# 1. Coste del terreno: 100x
# 2. Coste del panel solar: 250x
# 3. Coste de mantenimiento: 100.000 + 100x
# Coste total: 100x + 250x + 100.000 + 100x = 450x + 100.000
# """
# response = get_completion(prompt)
# print(response)

# Nótese que la solución del estudiante no es correcta. #
# Podemos arreglar esto instruyendo al modelo para que elabore su propia solución primero.

# prompt = f"""
# Su tarea es determinar si la solución del estudiante \
# es correcta o no.
# Para resolver el problema haz lo siguiente:
# - Primero, elabora tu propia solución al problema. 
# - Después compara tu solución con la del alumno \ 
# y valora si la solución del alumno es correcta o no. 
# No decidas si la solución del alumno es correcta hasta que 
# que hayas resuelto el problema tú mismo.

# Utiliza el siguiente formato:
# Pregunta:
# ```
# pregunta aquí
# ```
# Solución del alumno:
# ```
# solución del estudiante aquí
# ```
# Solución real:
# ```
# pasos para elaborar la solución y su solución aquí
# ```
# ¿Es la solución del alumno la misma que la solución real \
# que acaba de calcular:
# ```
# sí o no
# ```
# Calificación del alumno:
# ```
# correcta o incorrecta
# ```

# Pregunta:
# ```
# Estoy construyendo una instalación de energía solar y necesito ayuda para
# a calcular los costes. 
# - El terreno cuesta 100 $ / pie cuadrado
# - Puedo comprar paneles solares por $ 250 / pie cuadrado
# - He negociado un contrato de mantenimiento que me costará
# 100 mil dólares al año, y un adicional de 10 dólares por pie cuadrado.
# pie cuadrado
# ¿Cuál es el coste total para el primer año de operaciones /
# en función del número de pies cuadrados.
# ``` 
# Solución del alumno:
# ```
# Sea x el tamaño de la instalación en pies cuadrados.
# Costes:
# 1. Coste del terreno: 100x
# 2. Coste del panel solar: 250x
# 3. Coste de mantenimiento: 100.000 + 100x
# Coste total: 100x + 250x + 100.000 + 100x = 450x + 100.000
# ```
# Solución real:
# """
# response = get_completion(prompt)
# print(response)


######################################################################################################

# fact_sheet_chair = """
# VISIÓN GENERAL
# - Parte de una hermosa familia de muebles de oficina de inspiración de mediados de siglo, 
# que incluye archivadores, escritorios, librerías, mesas de reuniones y mucho más.
# - Varias opciones de color de carcasa y acabados de base.
# - Disponible con respaldo de plástico y tapizado frontal (SWC-100) 
# o tapizado completo (SWC-110) en 10 opciones de tela y 6 de piel.
# - Las opciones de acabado de la base son: acero inoxidable, negro mate 
# blanco brillante o cromado.
# - La silla está disponible con o sin reposabrazos.
# - Adecuada para uso doméstico o profesional.
# - Apta para uso contractual.

# CONSTRUCCIÓN
# - Base de aluminio plastificado de 5 ruedas.
# - Ajuste neumático de la silla para subir/bajar fácilmente.

# DIMENSIONES
# - ANCHO 53 CM | 20.87"
# - PROFUNDIDAD 51 CM | 20.08"
# - ALTURA 80 CM | 31.50"
# - ALTURA DEL ASIENTO 44 CM | 17.32"
# - PROFUNDIDAD DEL ASIENTO 41 CM | 16.14"

# OPCIONES
# - Opciones de ruedas para suelo blando o duro.
# - Dos opciones de densidades de espuma de asiento: 
#  media (1,8 lb/ft3) o alta (2,8 lb/ft3)
# - Reposabrazos de PU de 8 posiciones o sin brazos 

# MATERIALES
# CARCASA BASE DESLIZANTE
# - Aluminio fundido con revestimiento de nailon modificado PA6/PA66.
# - Grosor de la carcasa: 10 mm.
# ASIENTO
# - Espuma HD36

# PAÍS DE ORIGEN
# - Italia
# """

# prompt = f"""
# Su tarea consiste en ayudar a un equipo de marketing a crear una 
# descripción de un producto para un sitio web 
# en una ficha técnica.

# Escriba una descripción del producto basada en la información 
# proporcionada en la ficha técnica delimitada por 
# triple punto y coma.

# Ficha técnica: ```{fact_sheet_chair}```
# """
# response = get_completion(prompt)
# print(response)


# prompt = f"""
# Su tarea consiste en ayudar a un equipo de marketing a crear una 
# descripción de un producto para un sitio web 
# en una ficha técnica.

# Escriba una descripción del producto basada en la información 
# proporcionada en la ficha técnica delimitada por 
# tres puntos suspensivos.

# La descripción está destinada a minoristas de muebles, 
# por lo que debe ser de carácter técnico y centrarse en los 
# materiales con los que está fabricado el producto.

# Al final de la descripción, incluya cada uno de los 7 caracteres 
# ID del producto en la especificación técnica.

# Utilice un máximo de 50 palabras.

# Especificaciones técnicas: ```{fact_sheet_chair}```
# """
# response = get_completion(prompt)
# print(response)

# aviso = f"""
# Su tarea es ayudar a un equipo de marketing a crear un
# descripción para un sitio web minorista de un producto basado
# en una ficha técnica.

# Escriba una descripción del producto basada en la información.
# previsto en las especificaciones técnicas delimitadas por
# triples tildes.

# La descripción está destinada a minoristas de muebles,
# por lo que debe ser de naturaleza técnica y centrarse en la
# materiales con los que está construido el producto.

# Al final de la descripción, incluya cada 7 caracteres
# Identificación del producto en la especificación técnica.

# Después de la descripción, incluya una tabla que dé la
# dimensiones del producto. La tabla debe tener dos columnas.
# En la primera columna incluya el nombre de la dimensión.
# En la segunda columna incluya las medidas en pulgadas solamente.

# Asigne a la tabla el título 'Dimensiones del producto'.

# Formatee todo como HTML que se pueda usar en un sitio web.
# Coloque la descripción en un elemento <div>.

# Especificaciones técnicas: ```{fact_sheet_chair}```
# """

# response = get_completion(prompt)
# print (response)
# display(HTML(response))