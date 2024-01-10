import asyncio
import random

from PIL import ImageTk
import PySimpleGUI as sg

from bing_api import BingAPI
from config import Config
from openai_api import OpenAIBOT


class Interface:
    def __init__(self):
        self.config = Config()
        self.bing_api = BingAPI()
        self.imagens_baixadas_lazy = []
        self.personagem_lazy = None
        self.idade_lazy = None
    
    @staticmethod
    def atualiza_imagem(janela, key, imagem):
        imagem.thumbnail((500, 500))
        janela[key].update(data=ImageTk.PhotoImage(imagem))

    async def baixar_imagens_lazy(self, idade, personagem):
        if not self.imagens_baixadas_lazy or idade != self.idade_lazy or personagem != self.personagem_lazy:
            self.imagens_baixadas_lazy = await asyncio.to_thread(self.bing_api.obter_imagem, idade, personagem)
            self.personagem_lazy = personagem
            self.idade_lazy = idade
        else:
            random.shuffle(self.imagens_baixadas_lazy)

    async def main(self):
        sg.set_options(font=("Arial", 11))
        layout = [
            [sg.Text("Qual o seu nome? "),
             sg.Input(key="-NOME-", size=(30, 1))],
            [sg.Text("Qual a sua idade? "),
             sg.Input(key="-IDADE-", size=(30, 1))],
            [sg.Text("Onde você mora? "),
             sg.Input(key="-MORADIA-", size=(30, 1))],
            [sg.Text("Como você está se sentindo agora? "),
             sg.Input(key="-HUMOR-", size=(30, 1))],
            [sg.Text("Com quem você quer falar? "),
             sg.Input(key="-PERSONAGEM-", size=(30, 1))],
            [sg.Text("Escreva a mensagem (diga ou pergunte algo): "),
             sg.Input(key="-MENSAGEM-", size=(50, 15), expand_x=True)],
            [sg.Column([[sg.Button("Enviar mensagem ou pergunta"),
                         sg.Button("Limpar")]], justification="left")],
            [sg.Image(key="-IMAGEM1-", size=(500, 500)),
             sg.Output(key=("-RESPOSTA-"), size=(40, 30)),
             sg.Image(key="-IMAGEM2-", size=(500, 500))],
            [sg.Column([[sg.Button("Sair")]], justification="right")],
        ]
        janela = sg.Window("Personagem Virtual - by René - Versão ALPHA3.1", layout, resizable=True)
        historico = ""
        while True:
            try:
                evento, valores = janela.read(timeout=500)
                if evento == sg.WIN_CLOSED or evento == "Sair":
                    break
                elif evento == "Enviar mensagem ou pergunta":
                    nome = valores["-NOME-"]
                    idade = int(valores["-IDADE-"])
                    moradia = valores["-MORADIA-"]
                    humor = valores["-HUMOR-"]
                    personagem = valores["-PERSONAGEM-"]
                    mensagem = valores["-MENSAGEM-"]
                    tarefa_imagens = asyncio.create_task(self.baixar_imagens_lazy(idade, personagem))
                    instancia_conversa = OpenAIBOT(nome, idade, moradia, humor, personagem, mensagem)
                    resposta = instancia_conversa.enviar_prompt(historico)
                    historico += f" {nome} diz e/ou fez: {mensagem} e {personagem} respondeu e/ou fez, então {nome} diz e/ou faz:"
                    janela["-RESPOSTA-"].print(f"{personagem.upper()}:\n{resposta}\n\n")
                    await tarefa_imagens
                    for indice, imagem in enumerate(self.imagens_baixadas_lazy):
                        self.atualiza_imagem(janela, f"-IMAGEM{indice+1}-", imagem)
                        if indice >= 1:
                            break
                    janela.refresh()
            except Exception as erro:
                janela.close()
                sg.popup(f"Erro: {erro}", title="Erro Fatal")
                break
        janela.close()
    
    async def run(self):
        await self.main()


async def run_interface():
    interface = Interface()
    await interface.run()