import json

import openai
import os
import tiktoken.core
from tiktoken import get_encoding

from helpers.db_helper import DBHelper


class GPTHelper:
    def __init__(self):
        openai.api_key = 'API_KEY'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _ocultar_informacoes_sensiveis(self, data):

        if type(data) is str:
            data = json.loads(data)

        # Ocultar informações dos NPCs
        for npc in data.get("npcs", []):
            campos_sensiveis_npc = [
                "historia",
                "relacionamentos",
                "motivacoes",
                "honestidade",
                "habilidadesEspeciais",
                "opinioes",
                "localizacaoDuranteCrime",
                "culpado"
            ]
            for campo in campos_sensiveis_npc:
                npc.pop(campo, None)

        # Remover a solução inteira
        data.pop("solucao", None)

        return data

    def gerar_resumo(self, caso_json):

        system_prompt = f"""
        Voce é um policial que esta ajudando o detetive no caso
        voce ira receber um Json com os dados do caso e deverar criar um resumo do caso
        voce nao deve passar nenuma infomaçao que nao esteja no json
        faça um resumo do caso como se fosse um relatorio para o seu chefe
        utilize paragrafos organizar 
        Envie em formato de texto
        """

        json_suprimido = self._ocultar_informacoes_sensiveis(caso_json)


        messages = [{
            "role": "system",
            "content": str(system_prompt)
        }, {
            "role": "user",
            "content": str(caso_json)
        }]

        responses = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1024,
            temperature=0.8,
            stream=True,
        )

        return responses

    def gerar_prompt_suspeito(self, npc_id, caso_json):
        with DBHelper() as db:
            npc = db.get_json_npc(npc_id)
            npc_json = json.loads(npc[0])
            prompt = f"""
                Voce é um NPC de um jogo de investigação. Nao saia do seu personagem. Caso nescesario voce pode mentir.
                você esta sendo investigado por um crime.
                voce esta na sala de interrogatorio e o detetive esta te fazendo algumas perguntas.
                as informaçoes que voce tem sao:
                
                """

            prompt += json.dumps(npc_json, indent=4)

        return prompt

    def create_case(self):
        print("Creating case...")


        prompt = """Crie um caso completo de investigação para um jogo baseado em texto, incluindo NPCs, a situação do crime, pistas, interações e a solução do caso. Por favor, siga o seguinte formato JSON:
    
            {
              "caso": "Nome do caso",
              "descricao": "Descrição do caso",
              "npcs": [
                {
                  "nome": "Nome do NPC",
                  "historia": "História de fundo",
                  "relacionamentos": {
                    "amigos": ["Nome", "Nome"],
                    "inimigos": ["Nome"]
                  },
                  "ocupacao": "Profissão ou papel",
                  "personalidade": "Descrição breve",
                  "motivacoes": "Objetivos ou desejos",
                  "honestidade": "Nível de honestidade",
                  "habilidadesEspeciais": "Habilidades ou conhecimentos notáveis",
                  "opinioes": {
                    "NPC1": "Opinião sobre o NPC1",
                    "NPC2": "Opinião sobre o NPC2"
                  },
                  "localizacaoDuranteCrime": "Localização durante o evento principal",
                  "culpado": true/false
                },
                ... (outros 4 NPCs )
              ],
              "pistas": [
                {
                  "descricao": "Pista 1",
                  "origem": "Origem da pista",
                  "relevancia": "Relevância da pista para o caso"
                },
                ...
              ],
              "eventos": [
                {
                  "descricao": "Evento 1",
                  "participantes": ["Nome", "Nome"],
                  "momento": "Momento em relação ao crime"
                },
                ...
              ],
              "localizacoes": [
                {
                  "nome": "Nome do local",
                  "descricao": "Descrição do local",
                  "importancia": "Importância do local para o caso"
                },
                ...
              ],
              "solucao": {
                "resumo": "Resumo da solução do caso",
                "culpado": "Nome do culpado",
                "provas": ["Pista relevante 1", "Depoimento relevante", ...]
              }
            }
            """

        content = [
            {
                "role": "system",
                "content": str(prompt)
            }
        ]
        print("Sending prompt to GPT-4...")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=content,
            max_tokens=3000,
            temperature=0.8,
            stream=True,
            stop=["\n\n"]
        )
        text = ""
        for response in response:
            if response.choices[0].finish_reason:
                break
            text += response.choices[0].delta.content

        print("Received response from GPT-4.")

        return text

    def _count_tokens(self, text, model_name="gpt-3.5-turbo"):
        # encoding = get_encoding(model_name)
        # token_count = len(encoding.encode(text))
        # return token_count

        return len(text.split(" "))

    def _adjust_conversation_for_tokens(self, messages, new_message, max_tokens=4096):
        """
        Ajusta a lista de mensagens para garantir que ela, junto com a nova mensagem, não exceda max_tokens.
        Remove as mensagens mais antigas se necessário, mas mantém a mensagem 'system'.
        """
        total_tokens = self._count_tokens(new_message['content'])
        adjusted_messages = messages.copy()

        for message in reversed(adjusted_messages[1:]):  # Ignora a primeira mensagem (system)
            total_tokens += self._count_tokens(message['content'])

            if total_tokens > max_tokens:
                # Remova a mensagem mais antiga (após a mensagem 'system')
                total_tokens -= self._count_tokens(adjusted_messages.pop(1)['content'])
            else:
                break

        adjusted_messages.append(new_message)
        return adjusted_messages

    def chat(self, historic, message, functions=None):
        if functions is None:
            functions = []

        prompt = self._adjust_conversation_for_tokens(historic, message, max_tokens=512)

        print("Sending prompt to GPT-4...")
        print(prompt)
        responses = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            max_tokens=1024,
            temperature=1.4,
            stream=True,
        )

        return responses
