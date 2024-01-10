import openai
import os
import tkinter as tk
from tkinter import ttk

#import prompt_toolkit

#1000 tokens son 750 palabras por lo que 100 token deberían ser 75 palabras aproximadamente
# Configurar la API key de OpenAI
openai.api_key = os.getenv("MY_OPENAI_SECRET")


def obtener_respuesta(prompt):
    
    respuesta = openai.Completion.create(
        engine="text-davinci-003",  # Seleccionar el motor del modelo
        prompt=prompt,     # La pregunta o texto de entrada
        max_tokens=1500,     # Longitud máxima de la respuesta
        n=1,               # Generar solo una respuesta
        stop=None,         # No detener la respuesta en ningún token específico
        temperature=0,   # Controlar la "creatividad" de la respuesta
    )
    # Retornar la respuesta generada por el modelo
    return respuesta.choices[0].text.strip()

def enviar_mensaje():
    mensaje = entrada_texto.get()
    mensaje = "Dame información acerca de " + mensaje
    respuesta = obtener_respuesta(mensaje)
    if isinstance(salida_texto, tk.Text):
        salida_texto.delete(0, 'end')
    mostrar_respuesta(respuesta)
    

# Función para mostrar la respuesta en la interfaz gráfica
def mostrar_respuesta(respuesta):
    salida_texto.configure(state="normal")
    salida_texto.insert(tk.END, respuesta + "\n")
    salida_texto.configure(state="disabled")
    entrada_texto.delete(0, tk.END)

# Ventana principal
ventana = tk.Tk()
ventana.title("Prueba ChatGPT")

# Cuadro de texto de entrada
entrada_texto = ttk.Entry(ventana, width=50)
entrada_texto.pack(padx=10, pady=10)

# Botón de enviar
boton_enviar = ttk.Button(ventana, text="Enviar", command=enviar_mensaje)
boton_enviar.pack(padx=10, pady=5)

# Cuadro de texto de salida
salida_texto = tk.Text(ventana, width=50, height=20)
salida_texto.configure(state="disabled")
salida_texto.pack(padx=10, pady=10)

# Iniciar la aplicación en bucle
ventana.mainloop()


