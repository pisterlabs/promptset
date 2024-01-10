import openai

class ChatAssistant:
    def __init__(self, openai_api_key):
        openai.api_key = openai_api_key
        self._historico_mensagens = []  # Atributo protegido para armazenar o histórico de mensagens

    @property
    def historico_mensagens(self):
        return self._historico_mensagens

    @historico_mensagens.setter
    def historico_mensagens(self, value):
        self._historico_mensagens = value
        print("Conteúdo do histórico de mensagens:")
        for mensagem in self._historico_mensagens:
            print(f"Role: {mensagem['role']}, Content: {mensagem['content']}")

    def adicionar_mensagem(self, role, content):
        self.historico_mensagens.append({"role": role, "content": content})

    def enviar_solicitacao(self):
        total_tokens = sum(len(msg['content'].split()) for msg in self.historico_mensagens)
        max_tokens = 369
        tokens_excedentes = total_tokens - 3369 + max_tokens

        if tokens_excedentes > 0:
            # Agrupa as mensagens em grupos menores que não excedam o limite de tokens
            grupos_mensagens = []
            grupo_atual = []
            total_tokens_grupo_atual = 0
            for mensagem in self.historico_mensagens:
                tokens_mensagem = len(mensagem['content'].split())
                if total_tokens_grupo_atual + tokens_mensagem <= max_tokens:
                    grupo_atual.append(mensagem)
                    total_tokens_grupo_atual += tokens_mensagem
                else:
                    grupos_mensagens.append(grupo_atual)
                    grupo_atual = [mensagem]
                    total_tokens_grupo_atual = tokens_mensagem
            if grupo_atual:
                grupos_mensagens.append(grupo_atual)

            # Envia os grupos de mensagens como uma lista de listas
            resultado = ''
            for grupo in grupos_mensagens:
                result = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo-0301',
                    messages=grupo,
                    max_tokens=max_tokens
                )
                resultado += result['choices'][0]['message']['content'].strip()

            return resultado

        result = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0301',
            messages=self.historico_mensagens,
            max_tokens=max_tokens
        )
        return result['choices'][0]['message']['content'].strip()

    def remover_excesso_tokens(self):
        total_tokens = sum(len(msg['content'].split()) for msg in self.historico_mensagens)
        if total_tokens > 3369:
            tokens_excedentes = total_tokens - 3369
            count = 0
            indice_remocao = -1
            for i in range(len(self.historico_mensagens) - 1, -1, -1):
                count += len(self.historico_mensagens[i]['content'].split())
                if count > tokens_excedentes:
                    indice_remocao = i
                    break
            if indice_remocao >= 4:
                self.historico_mensagens = self.historico_mensagens[:indice_remocao + 1]
