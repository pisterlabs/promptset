import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import openai

# Configura tu clave de API de GPT-3
api_key = 'tu_clave_de_api_aqui'

# Crea una figura y un gráfico en blanco
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4, 5], [10, 5, 8, 3, 6])  # Agrega tus datos aquí

# Genera la descripción del gráfico utilizando GPT-3
descripcion = "Genera una descripción para este gráfico:"

openai.api_key = api_key
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=descripcion,
    max_tokens=50  # Ajusta la longitud máxima de la descripción
)

# Obtiene la descripción generada por GPT-3
descripcion_generada = response.choices[0].text

# Muestra la descripción en la consola
print(descripcion_generada)

# Guarda el gráfico en un archivo o muéstralo en una ventana emergente según tus necesidades
canvas = FigureCanvasAgg(fig)
canvas.print_figure('mi_grafico.png', dpi=100)
plt.show()
