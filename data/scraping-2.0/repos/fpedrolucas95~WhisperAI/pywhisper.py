import sys
import openai
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk, Text

app_version = 0.3  
app_date = "2 de outubro de 2023"

def transcrever_audio():
    caminho_audio = campo_arquivo.get()
    chave_api = campo_chave_api.get()

    if not chave_api:
        messagebox.showerror("Erro", "Chave da API OpenAI não fornecida!")
        return

    openai.api_key = chave_api
    idioma = selecao_idioma.get()
    
    with open(caminho_audio, "rb") as arquivo_audio:
        transcript = openai.Audio.transcribe(
            file=arquivo_audio,
            model="whisper-1",
            response_format="text",
            language=idioma
        )
    
    area_texto.delete('1.0', tk.END) 
    area_texto.insert(tk.END, transcript)  

def selecionar_arquivo():
    caminho_arquivo = filedialog.askopenfilename(filetypes=[("Arquivos de áudio", "*.mp3")])
    campo_arquivo.delete(0, tk.END)
    campo_arquivo.insert(0, caminho_arquivo)

def mostrar_sobre():
    chave_presente = "Sim" if campo_chave_api.get() else "Não"
    cor_chave = "green" if campo_chave_api.get() else "red"
    sobre_janela = tk.Toplevel()
    sobre_janela.title("Sobre")
    tk.Label(sobre_janela, text=f"Nome da Aplicação: Interface Desktop para o Whisper AI", justify=tk.LEFT).pack()
    tk.Label(sobre_janela, text=f"Versão da Aplicação: {app_version}", justify=tk.LEFT).pack()
    tk.Label(sobre_janela, text=f"Versão do Python: {sys.version.split()[0]}", justify=tk.LEFT).pack()
    tk.Label(sobre_janela, text=f"Versão da API do OpenAI: {openai.__version__}", justify=tk.LEFT).pack()
    tk.Label(sobre_janela, text=f"Chave OpenAI: {chave_presente}", justify=tk.LEFT, foreground=cor_chave).pack()
    tk.Label(sobre_janela, text=f"Link para o GitHub: https://github.com/fpedrolucas95", justify=tk.LEFT).pack()
    tk.Button(sobre_janela, text="Sair", command=sobre_janela.destroy).pack()

def mostrar_config():
    campo_chave_api.set(simpledialog.askstring("ㅤ", "Chave OpenAI:"))

def salvar_transcricao():
    texto_transcricao = area_texto.get('1.0', tk.END).strip()
    if texto_transcricao:
        caminho_salvar = filedialog.asksaveasfilename(defaultextension=".txt",
                                                      filetypes=[("Arquivos de texto", "*.txt")])
        if caminho_salvar:
            with open(caminho_salvar, "w", encoding='utf-8') as arquivo_transcricao:
                arquivo_transcricao.write(texto_transcricao)
    else:
        messagebox.showinfo("Informação", "Nenhuma transcrição disponível para salvar.")

janela = tk.Tk()
janela.title("Transcrição de Áudio")
janela.resizable(0, 0) 

menu = tk.Menu(janela)
janela.config(menu=menu)
menu_config = tk.Menu(menu)
menu.add_cascade(label="Configuração", menu=menu_config)
campo_chave_api = tk.StringVar()
menu_config.add_command(label="Chave OpenAI", command=mostrar_config)
menu_config.add_command(label="Sobre", command=mostrar_sobre)

tk.Label(janela, text="Áudio:").grid(row=0, column=0, sticky=tk.W)
campo_arquivo = tk.Entry(janela, width=50)
campo_arquivo.grid(row=0, column=1)
tk.Button(janela, text="Selecionar Arquivo", command=selecionar_arquivo).grid(row=0, column=2)

tk.Label(janela, text="Idioma:").grid(row=1, column=0, sticky=tk.W)
opcoes_idioma = ["pt", "en", "es"]
selecao_idioma = ttk.Combobox(janela, values=opcoes_idioma)
selecao_idioma.grid(row=1, column=1)
selecao_idioma.set("pt")

tk.Button(janela, text="Transcrever Áudio", command=transcrever_audio).grid(row=1, column=2)

area_texto = Text(janela, width=60, height=15)
area_texto.grid(row=2, columnspan=3, pady=10)

tk.Button(janela, text="Salvar Transcrição", command=salvar_transcricao).grid(row=3, columnspan=3)

janela.mainloop()
