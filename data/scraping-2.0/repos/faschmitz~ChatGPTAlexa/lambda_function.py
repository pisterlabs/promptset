import logging
import os
import sys
import re

import openai
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.utils import is_intent_name, is_request_type
from ask_sdk_model import Response
from ask_sdk_model.interfaces.display import PlainText, TextContent
from ask_sdk_model.ui import SimpleCard

# Configurar chave de API do OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]

def get_chat_gpt_response(prompt):
    messages = [{"role": "system", "content": "Por favor, responda de forma clara, concisa e conversacional em linguagem natural portugues."},
                {"role": "system", "content": "Responda tudo numa mesma linha."},
                {"role": "user", "content": prompt}]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].message["content"].strip()


def escape_ssml(text):
    """Substitui caracteres problemáticos no SSML."""
    return re.sub(r'([&<>])', lambda m: {'&': '&amp;', '<': '&lt;', '>': '&gt;'}[m.group(1)], text)

# Request Handler para a intenção ChatGPT
class ChatGPTIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("ChatGPTIntent")(handler_input)

    def handle(self, handler_input):
        # Obter a mensagem do usuário a partir do slot 'message'
        message = handler_input.request_envelope.request.intent.slots["message"].value

        # Obter resposta do ChatGPT
        chat_gpt_response = get_chat_gpt_response(message)

        # Escapar caracteres SSML problemáticos
        speech_text = escape_ssml(chat_gpt_response)
        
        # Adicionar mensagem adicional solicitando nova pergunta ou ação
        #speech_text += " Se você tiver mais perguntas ou precisar de ajuda, por favor, me avise."

        # Adicionar reprompt com atraso de 10 segundos
        reprompt_text = "<break time='10s'/> Encerrando a sessão devido à inatividade. Se precisar de mais ajuda, sinta-se à vontade para invocar a skill novamente."
    
        
        # Responder ao usuário
        handler_input.response_builder.speak(speech_text).ask(reprompt_text).set_card(SimpleCard("ChatGPT", speech_text))
        return handler_input.response_builder.response
    
# Adicione o LaunchRequestHandler
class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        speech_text = "Bem-vindo ao Chat. Como posso ajudá-lo hoje?"
        reprompt_text = "Por favor, diga o que você gostaria de fazer."
        handler_input.response_builder.speak(speech_text).ask(reprompt_text).set_should_end_session(False)
        
        return handler_input.response_builder.response
    
# Request Handler para o encerramento devido à inatividade
class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        return handler_input.response_builder.response    

# Adicione o HelpIntentHandler
class HelpIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        speech_text = "Você pode me fazer uma pergunta ou iniciar uma conversa."
        handler_input.response_builder.speak(speech_text).set_should_end_session(False)
        return handler_input.response_builder.response

# Adicione o FallbackIntentHandler
class FallbackIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        speech_text = "Desculpe, não entendi o que você disse. Por favor, tente novamente."
        handler_input.response_builder.speak(speech_text).ask(speech_text).set_should_end_session(False)
        return handler_input.response_builder.response        

# Função Lambda principal
sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(ChatGPTIntentHandler())

def lambda_handler(event, context):
    return sb.lambda_handler()(event, context)


def main(busca):
    print(get_chat_gpt_response(busca))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        argumento = sys.argv[1]  # Obtém o primeiro argumento (o segundo item na lista sys.argv)
        main(argumento)  # Chama a função main passando o argumento
    else:
        print("Nenhum argumento foi passado.")
