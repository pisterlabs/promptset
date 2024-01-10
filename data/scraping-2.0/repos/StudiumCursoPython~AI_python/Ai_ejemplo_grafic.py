""" 
Curso Python empresa de 'Lenguaje de Programación Python'

Autor: José Antonio Calvo López

Fecha: Noviembre 2023

"""

from tkinter import scrolledtext, messagebox
import tkinter as tk
import openai


# Configura tu clave de API aquí
openai.api_key = "API En mi Drive"

# Configuración inicial
context = {"role": "system", "content": "Eres un asistente muy útil."}
messages = [context]

# Funciones
def enviar_consulta():
    content = entry.get()

    if content == "new":
        actualizar_texto("🆕 Nueva conversación creada")
        global messages
        messages = [context]
        entry.delete(0, tk.END)
        return

    messages.append({"role": "user", "content": content})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages)
        response_content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": response_content})
        actualizar_texto(f"> {response_content}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

    entry.delete(0, tk.END)

def actualizar_texto(texto):
    text_area.insert(tk.INSERT, f"{texto}\n")
    text_area.yview(tk.END)

def salir():
    if messagebox.askyesno("Salir", "¿Estás seguro que deseas salir?"):
        root.destroy()

# Configuración de la ventana principal
root = tk.Tk()
root.geometry("850x450")
root.resizable(False,False)
root.title("AI con ChatGPT GUI")

LabelFrame = tk.Label(root, text="¿Qué quieres saber?")
LabelFrame.pack()

# Área de entrada de texto
entry = tk.Entry(root, width=100)
entry.pack()

LabelFrame = tk.Label(root, text="* Escribre 'new' para nueva conversación.", font=("Helvetica", 7))
LabelFrame.pack()

# Botón para enviar la consulta
send_button = tk.Button(root, text="Enviar", command=enviar_consulta)
send_button.pack()

# Área de visualización de respuestas
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=20)
text_area.pack()

# Botón de salida
exit_button = tk.Button(root, text="Salir", command=salir)
exit_button.pack()

# Iniciar la GUI
root.mainloop()
