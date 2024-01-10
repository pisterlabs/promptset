import os
from openai import OpenAI

key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=key)

text = """
La nube pública es un modelo de despliegue que contempla el uso de una infraestructura compartida que ofrecen los proveedores de nube, en este caso, Microsoft. Este modelo de despliegue es el más común en el cómputo en la nube. La nube pública es un ambiente multitenant. Este término en inglés se refiere a que podemos tener un gran número de usuarios que estén usando la misma infraestructura; de hecho, tenant significa inquilino en inglés, por lo que entonces podemos decir que es un ambiente multiinquilino. Este modelo permite a las empresas adoptar rápidamente la nube, ya que el proveedor ya tiene toda la infraestructura preparada y configurada de antemano. Este modelo tiene muchas ventajas, entre las que podemos destacar la escalabilidad prácticamente ilimitada; esto obedece a que ya están disponibles para nosotros los recursos de cómputo que el proveedor tiene a nuestra disposición. La nube pública nos brinda una agilidad única para poder adaptarnos rápidamente a los cambios en el mercado donde se encuentre nuestra empresa. Además, gracias a su modelo de pago por uso, podemos controlar los costos de las soluciones de software que implementemos. También, la nube pública ofrece una gran cantidad de funcionalidades con muy poco esfuerzo y conocimientos técnicos, principalmente, claro, si usamos los servicios de plataforma como servicio. Y finalmente, nos ofrece un modelo de autoservicio. ¿Qué significa esto? Significa que en cualquier momento que nosotros lo necesitemos, podemos hacer uso de una gran cantidad de recursos de cómputo y servicios, los cuales ya están listos para ser desplegados y usados. Si bien la nube pública tiene bastantes beneficios, mencionemos algunas desventajas que tiene este modelo. La primera desventaja es que tienes que ceder el control de la infraestructura a un tercero, en este caso, Microsoft; esto significa que el proveedor tiene todo el poder de decisión con respecto al hardware. Adicionalmente, algunas empresas e instituciones gubernamentales, están regidas por regulaciones y leyes que les impiden el uso de una nube pública, sobre todo si esa nube está en un país diferente. Sin embargo, cada vez son más flexibles las leyes y gobiernos, y ya se están alineando a los cambios tecnológicos de nuestros tiempos.
"""

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        { "role":"system", "content": "Eres un asistente muy útil." },
        { "role":"user", "content": 
        f"""
        Realiza las siguientes tareas:
        1. Resume en una sola frase el texto que está entre 3 asteriscos.
        2. Traduce la frase al Inglés.
        3. Debajo del texto, crea una lista en Inglés con los 5 temas principales del texto original.

        ***
        {text}
        ***
        """ }
    ]
)

print(completion.choices[0].message.content)