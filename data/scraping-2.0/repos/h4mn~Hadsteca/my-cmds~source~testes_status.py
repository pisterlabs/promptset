# Ferramenta para Monitorar o Status do Processo de uma Tarefa
#
# - Mostra o status de um processo;
# - Exibe o status em uma barra de progresso horizontal;
# - Atualiza o status a cada 1 segundo e é mostrado em um Progress Circular;
# - Mostra o título da Tarefa na barra de título da janela;
# - Exibe o status em um rótulo.
#
# O código inclui uma classe para um componente ProgressCircular que desenha um círculo indicando o progresso a cada segundo, e uma classe Aplicativo que lê um arquivo e atualiza o progresso na barra de progresso e no rótulo. O programa principal cria uma instância da classe Aplicativo e inicia o loop principal do Tkinter. O código também lida com redimensionamento da janela e atualiza a largura da barra de progresso conforme necessário.
#
# ToDo: Fazer melhorias nas classes
# - Separar constantes para reutilização

import tkinter as tk
from tkinter import ttk
import re
import time
import math
import openai
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente
load_dotenv()

#
# Classe do GPT
#
class GPT:
  # Construtor
  def __init__(self):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    self.engine = "text-davinci-003"
    self.temperature = 0
    self.max_tokens = 100
    #self.stop = ["\n", "Q:"]
    self.echo = False
    self.persona = """
    Converta este markdown para texto com no máximo 12 palavras, no gerúndio.
    """

  # Função para gerar o texto
  def gerar_texto(self, prompt):
    response = openai.Completion.create(
      engine=self.engine,
      prompt=f"{self.persona}\n---\n{prompt}\n",
      temperature=self.temperature,
      max_tokens=self.max_tokens,
      #stop=self.stop,
    )
    if self.echo:
      print(response)
    return response.choices[0].text.strip()


# 
# Classe do componente ProgressCircular
# 
class ProgressCircular(tk.Canvas):
  def __init__(self, parent, size=100, thickness=0, progress=0, countdown_value=0, *args, **kwargs):
    tk.Canvas.__init__(self, parent, width=size, height=size, highlightthickness=0, *args, **kwargs, )
    self.size = size
    self.thickness = thickness
    self.progress = progress
    self.countdown_value = countdown_value
    self.draw_progressbar()
    
  def draw_progressbar(self):
    self.delete("progress")
    angle = (self.progress / 100) * 360
    radians = math.radians(90 - angle)
    x = self.size / 2 + (self.size / 2 - self.thickness / 2) * math.cos(radians)
    y = self.size / 2 - (self.size / 2 - self.thickness / 2) * math.sin(radians)

    self.create_arc(
        self.thickness / 2,
        self.thickness / 2,
        self.size - self.thickness / 2,
        self.size - self.thickness / 2,
        start=90,
        extent=-angle,
        width=0,
        fill="#20a842",
        tags="progress",
    )

  def set_progress(self, progress):
    self.progress = progress
    self.draw_progressbar()

  def set_countdown_value(self, countdown_value):
    self.countdown_value = countdown_value


#
# Classe do aplicativo
#
class Aplicativo:
  def __init__(self, janela):
    # Inicializa GPT
    self.gpt = GPT()
    self.last_titulo = ""
    self.last_tarefa = ""

    # Define o título da janela
    self.tarefa_em_progresso = ""

    # Criação do frame
    self.frame = tk.Frame(janela)
    self.frame.pack(fill=tk.BOTH, expand=True)

    # Criação do componente ProgressCircular
    self.circular_progressbar = ProgressCircular(self.frame, size=100, thickness=10, progress=0, countdown_value=0)
    self.circular_progressbar.grid(row=0, column=0, rowspan=3, sticky=tk.EW, padx=5)

    # Criação do rótulo da tarefa
    self.rotulo_tarefa = tk.Label(self.frame, text="")
    self.rotulo_tarefa.grid(row=0, column=1, sticky=tk.EW, padx=5)

    # Criação do rótulo dos contadores
    self.rotulo = tk.Label(self.frame, text="")
    self.rotulo.grid(row=1, column=1, sticky=tk.EW, padx=5)

    # Criação da barra de progresso
    self.barra_progresso = ttk.Progressbar(self.frame, mode="determinate", length=100)
    self.barra_progresso.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)

    # Define o tamanho da coluna 1
    self.frame.columnconfigure(1, minsize=480)

    # Atualize o progresso circular
    self.update_countdown()

    # Atualize a barra de progresso
    self.atualizar_arquivo()

  # Função para ler o arquivo
  def ler_arquivo(self):
    #arquivo = r"Z:\Backup\_tmp\_md\tarefas\3058201.md"
    #arquivo = r"Z:\Backup\_tmp\_md\tarefas\20230502.1109\fix_iOS.md"
    #arquivo = "Z://Backup//_tmp//_md//tarefas//20230502.1109//fix_iOS.md"
    #arquivo = r"Z:\Backup\_tmp\_md\tarefas\20230505.1124\fix_Android.md"
    #arquivo = r"Z:\Backup\_tmp\_md\tarefas\3313401.md"
    arquivo = r"Z:\Backup\_tmp\_md\tarefas\3027901_rascunhos.md"

    # Monitoramento dos Testes: Padrão para encontrar os caracteres "[x]" e "[ ]"
    #tarefa_em_progresso = "Corrigindo validação de super admin na tela de Classificação Fiscal"
    #padrao_pendente = r"\[\s\]"
    #padrao_realizado = r"\[x\]"

    # Monitoramento dos RelDinTpLkp_ServicoProduto: Padrão para encontrar "\] Produto" e "\] Produto [^7]"    
    #padrao_pendente = r"\]\sProduto\n"
    #padrao_realizado = r"\]\sProduto\s\[\^7\]\n"

    # Monitoramento da tarefa 3313401: Padrão para encontrar checklists
    tarefa_em_progresso = "Revisão Dashboard Futura Util"
    # padrao_pendente = r"\-\s\[\s\]\s"
    # padrao_realizado = r"\-\s\[x\]\s"
    padrao_pendente = r"\-\s\[\s\]\s"
    padrao_realizado = r"\-\s\[x\]\s"

    qtd_char1 = 0
    qtd_char2 = 0
    primeira_linha = ""
    proxima_tarefa = ""
    proxima_tarefa_prefixo = "- [ ]"

    try:
        with open(arquivo, "r", encoding='utf-8') as f:
            linhas = f.readlines()
            primeira_linha = linhas[0].strip() if linhas else ""
            for linha in linhas:
              if linha.startswith(proxima_tarefa_prefixo):
                proxima_tarefa = linha.strip()
                break

            conteudo = "".join(linhas)
            qtd_char1 = len(re.findall(padrao_realizado, conteudo))
            qtd_char2 = len(re.findall(padrao_pendente, conteudo))
    except FileNotFoundError:
        print(f"Arquivo {arquivo} não encontrado.")
    
    total = qtd_char1 + qtd_char2
    return total, qtd_char1, qtd_char2, primeira_linha, proxima_tarefa

  # Função para atualizar a barra de progresso
  def atualizar_arquivo(self):
    total, qtd_char1, qtd_char2, titulo, tarefa = self.ler_arquivo()

    # Atualize a barra de progresso
    self.barra_progresso["value"] = qtd_char1
    self.barra_progresso["maximum"] = total

    progresso_percentual = (qtd_char1 / total) * 100 if total != 0 else 0
    self.rotulo.config(
      text=f"Verificados: {qtd_char1}, Restantes: {qtd_char2}, Total: {total}, Progresso: {progresso_percentual:.1f}%"
    )

    # Atualize a label da tarefa
    if tarefa != self.last_tarefa:
      self.last_tarefa = tarefa
      tarefa = self.gpt.gerar_texto(tarefa)
      self.rotulo_tarefa.config(text=f"Etapa da tarefa: {tarefa}")

    # Define o título da janela
    if titulo != self.last_titulo:
      self.last_titulo = titulo
      titulo = self.gpt.gerar_texto(titulo)
      janela.title(f"Tarefa atual: {titulo}")
  
  # Função para redimensionar a barra de progresso
  def on_resize(self, event):
    width = event.width
    self.barra_progresso.configure(length=width-110)
    self.rotulo.configure(wraplength=width-110)

  def update_countdown(self):
    if self.barra_progresso["value"] == self.barra_progresso["maximum"]:
      return

    countdown_value = self.circular_progressbar.countdown_value - 1
    if countdown_value < 0:
      countdown_value = 60

    progress = (60 - countdown_value) * (100 / 60)
    self.circular_progressbar.set_countdown_value(countdown_value)
    self.circular_progressbar.set_progress(progress)

    if countdown_value == 0:
      self.atualizar_arquivo()

    janela.after(1000, self.update_countdown)


#
# Função principal
#
if __name__ == "__main__":
    janela = tk.Tk()

    # Define a largura da janela
    janela.geometry("640x100+100+900")

    janela.resizable(True, False)

    # Define a toplevel da janela
    janela.attributes("-topmost", True)
    # Defina a cor transparente para a janela
    janela.attributes("-transparentcolor", "white")

    app = Aplicativo(janela)

    # Vincula a função on_resize ao evento de redimensionamento da janela
    janela.bind("<Configure>", app.on_resize)

    janela.mainloop()
