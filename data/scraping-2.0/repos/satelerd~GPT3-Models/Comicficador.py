"""
Este modelo tiene mucha importancia ya que fue la idea con la cual postule al API de OpenAI.

La idea es que un comicficador devuelva el texto entregado pero con una serie de modificaciones
como negrita, cursiva, destacado y cambio de color, fuente y tamaño de letra dependiendo del contexto.
De esta manera el texto sera mucho mas divertido e "interactivo" para el usuario.
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
    engine="davinci",
    prompt='Lo siguiente es un Comicficador, en el cual, dado un texto devolverá el texto Comicficado. En un texto Comicficado se cambia el tamaño y la negrita de frases o palabras, dependiendo de la importancia o el tono que tengan. De esta forma el texto se puede leer de forma más rápida y didáctica.\n# #= Texto grande\n** ** = Negrita\n\nTexto:\nSoy Ana, tengo veinte años, y estudio en la universidad, donde también trabajo como profesora de historia. Viví en un pueblito pequeño, con una vida tranquila, hasta que un día unos alienígenas llegaron y me secuestraron. No sé cómo se llamaban, pero sí que eran muy guapos.\n\nTexto Comicficado:\nSoy **Ana**, tengo **veinte años** , y estudio en la universidad, donde también trabajo como profesora de historia. Viví en un **pueblito pequeño**, con una vida tranquila, hasta que #un día unos **alienígenas** llegaron y me **secuestraron**#. No sé cómo se llamaban, pero sí que eran **muy guapos** .\n\n###\n\nTexto:\nEstabamos en plena guerra contra los sovieticos, y yo era un soldado más. Teniamos una vida precaria, pero valiente. Un día fui a una misión solo, y al llegar al punto de encuentro, me encontré con un hombre que me miraba fijamente. y que empezo a gritar: "¡No te muevas! ¡Soy un alienígena!". Me dio miedo, y empecé a correr. Pero me atrapó y me llevó a su nave. Donde me violó.\n\nTexto Comicficado:\nEstabamos en plena **guerra contra los sovieticos**, y yo era un soldado más. Teniamos una vida precaria, pero valiente. Un día **fui a una misión solo,** y al llegar al punto de encuentro, me encontré con un **hombre** que me miraba fijamente. y que **empezo a gritar: #"¡No te muevas! ¡Soy un alienígena!"#**. Me dio miedo, y **empecé a correr**. Pero me atrapó y me llevó a su nave. Donde me **violó**.\n\n###\n\nTexto:',
    temperature=0.67,
    max_tokens=64,
    top_p=1,
    frequency_penalty=0.24,
    presence_penalty=0.23,
)
