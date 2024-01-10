# -*- coding: utf-8 -*-

# This sample demonstrates handling intents from an Alexa skill using the Alexa Skills Kit SDK for Python.
# Please visit https://alexa.design/cookbook for additional examples on implementing slots, dialog management,
# session persistence, api calls, and more.
# This sample is built using the handler classes approach in skill builder.
import openai
import logging
import gettext
from openai import OpenAI
import os
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import (
    AbstractRequestHandler, AbstractRequestInterceptor, AbstractExceptionHandler)
import ask_sdk_core.utils as ask_utils
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
import config
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage

import json
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_to_dict
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder, 
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationSummaryBufferMemory


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
openai.api_key = os.environ['API_Key']

# openaiモジュールのインポートが成功したかどうかをチェック
# try:
#     # 例として、openaiのバージョンを取得して表示
#     openai_version = openai.__version__
#     print(f"OpenAI successfully imported. Version: {openai_version}")
# except AttributeError:
#     print("Failed to import OpenAI.")

class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""

    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        _ = handler_input.attributes_manager.request_attributes["_"]
        speak_output = "アスカだけど、なんなのよ全く"

        return (
            handler_input.response_builder
            .speak(speak_output)
            .ask(speak_output)
            .response
        )


class HelloWorldIntentHandler(AbstractRequestHandler):
    """Handler for Hello World Intent."""

    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("HelloWorldIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        _ = handler_input.attributes_manager.request_attributes["_"]
        speak_output = "ハイ"

        return (
            handler_input.response_builder
            .speak(speak_output)
            # .ask("add a reprompt if you want to keep the session open for the user to respond")
            .response
        )


class HelpIntentHandler(AbstractRequestHandler):
    """Handler for Help Intent."""

    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        _ = handler_input.attributes_manager.request_attributes["_"]
        speak_output = "どうしましたか？"

        return (
            handler_input.response_builder
            .speak(speak_output)
            .ask(speak_output)
            .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Single handler for Cancel and Stop Intent."""

    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        _ = handler_input.attributes_manager.request_attributes["_"]
        speak_output = "さようなら"

        return (
            handler_input.response_builder
            .speak(speak_output)
            .response
        )

class FallbackIntentHandler(AbstractRequestHandler):
    """Single handler for Fallback Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        logger.info("In FallbackIntentHandler")
        speech = "Hmm, I'm not sure. You can say Hello or Help. What would you like to do?"
        reprompt = "I didn't catch that. What can I help you with?"

        return handler_input.response_builder.speak(speech).ask(reprompt).response

class SessionEndedRequestHandler(AbstractRequestHandler):
    """Handler for Session End."""

    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response

        # Any cleanup logic goes here.

        return handler_input.response_builder.response


class IntentReflectorHandler(AbstractRequestHandler):
    """The intent reflector is used for interaction model testing and debugging.
    It will simply repeat the intent the user said. You can create custom handlers
    for your intents by defining them above, then also adding them to the request
    handler chain below.
    """

    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("IntentRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        _ = handler_input.attributes_manager.request_attributes["_"]
        intent_name = ask_utils.get_intent_name(handler_input)
        speak_output = "こんにちは"

        return (
            handler_input.response_builder
            .speak(speak_output)
            # .ask("add a reprompt if you want to keep the session open for the user to respond")
            .response
        )


class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors. If you receive an error
    stating the request handler chain is not found, you have not implemented a handler for
    the intent being invoked or included it in the skill builder below.
    """

    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> Response
        logger.error(exception, exc_info=True)
        _ = handler_input.attributes_manager.request_attributes["_"]
        speak_output = "例外です。"
        
        # 例外の詳細をログに記録
        logger.error(f"Error occurred: {exception}")

        return (
            handler_input.response_builder
            .speak(speak_output)
            .ask(speak_output)
            .response
        )


class LocalizationInterceptor(AbstractRequestInterceptor):
    """
    Add function to request attributes, that can load locale specific data
    """

    def process(self, handler_input):
        locale = handler_input.request_envelope.request.locale
        i18n = gettext.translation(
            'data', localedir='locales', languages=[locale], fallback=True)
        handler_input.attributes_manager.request_attributes["_"] = i18n.gettext

# The SkillBuilder object acts as the entry point for your skill, routing all request and response
# payloads to the handlers above. Make sure any new handlers or interceptors you've
# defined are included below. The order matters - they're processed top to bottom.
# class ChatGPTIntentHandler(AbstractRequestHandler):
#     openai.api_key = os.environ['API_Key']
#     conversation_history = []
#     def can_handle(self, handler_input):
#         return ask_utils.is_intent_name("ChatGPTIntent")(handler_input)

#     def handle(self, handler_input):
#         client = OpenAI(api_key=os.environ['API_Key'])  # OpenAIクライアントのインスタンスを作成
#         slots = handler_input.request_envelope.request.intent.slots
#         question = slots["question"].value
        
#         # selfを使ってconversation_historyにアクセス
#         #ChatGPTIntentHandler.conversation_history.append(f"ユーザー: {question}")
#         print('self1はOK')

#         #conversation_history.append(f"ユーザー: {question}")

#         #prompt = "以下は、ユーザーとAIとの会話です。\n\n"
#         # prompt = "\n"
#         # for message in ChatGPTIntentHandler.conversation_history:
#         #     prompt += f"{message}\n"

#         # prompt += "AI: "        
        
#         messages = [
#             {"role" : "system", "content":config.asuka_alexa_prompt
#             },
#             {"role" : "user", "content": question}
#         ]
#         print('self2はOK')

#         # res=openai.ChatCompletion.create(
#         #     model="gpt-3.5-turbo",
#         #     messages=messages,
#         #     max_tokens = 200
#         # )
        
#         response = client.chat.completions.create(
#             model="gpt-4-1106-preview",
#             #response_format={"type": "json_object"},
#             messages=messages,
#             max_tokens=1000
#         )

#         # speak_output = res['choices'][0]['message']['content']
#         speak_output = response.choices[0].message.content

#         speech_text = speak_output
        
#         print(f'sppech text:{speech_text}')
        
#         #answer = res['choices'][0]['message']['content']#.strip()
#         # selfを使ってconversation_historyにアクセス
#         #ChatGPTIntentHandler.conversation_history.append(f"AI: {speech_text}")
#         #conversation_history.append(f"AI: {answer}")
#         # if response["usage"]["total_tokens"] > 1500:
#         #     conversation_history.pop(0)
#         #     conversation_history.pop(0)

#         return (
#             handler_input.response_builder
#             .speak(speech_text)
#             .ask("まだなんか聞きたいわけ?")
#             .response
#         )

class LCChatGPTIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("LCChatGPTIntent")(handler_input)

    def handle(self, handler_input):
        #openai.api_key = os.environ['API_Key']
        chat = ChatOpenAI(temperature=1.0, openai_api_key=os.environ['API_Key'],model_name='gpt-4-1106-preview') # OpenAIクライアントのインスタンスを作成
        slots = handler_input.request_envelope.request.intent.slots
        question = slots["question"].value
        
        # セッション属性から会話履歴を取得
        session_attr = handler_input.attributes_manager.session_attributes
        if "conversation_history" not in session_attr:
            session_attr["conversation_history"] = []
        
        # selfを使ってconversation_historyにアクセス
        #ChatGPTIntentHandler.conversation_history.append(f"ユーザー: {question}")
        print('langcahin:self1はOK')
        
        # 会話履歴に現在の質問を追加
        session_attr["conversation_history"].append(f"ユーザー: {question}")
        
        # 会話の履歴を保持するためのオブジェクト
        memory = ConversationBufferMemory(return_messages=True)
        template = config.asuka_alexa_prompt
        print('langcahin:self2はOK')

        # res=openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=messages,
        #     max_tokens = 200
        # )
        
        # テンプレートを用いてプロンプトを作成
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(template),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
        # AIとの会話を管理
        conversation = ConversationChain(llm=chat, memory=memory, prompt=prompt)

        #会話
        response = conversation.predict(input=question)
        # chain = ConversationChain(
        #     llm=chat, prompt=prompt, memory=memory, verbose=True
        # )
        #response = chain.run(question)
        # 会話履歴をmemoryオブジェクトに保存
        memory.save_context({"input": question}, {"output": response})
        print(memory.load_memory_variables({}))
        
        
        # 会話履歴を更新
        session_attr["conversation_history"].append(f"AI: {response}")
        handler_input.attributes_manager.session_attributes = session_attr
        #handler_input.attributes_manager.save_persistent_attributes()
        
        # 会話履歴を出力
        print("現在の会話履歴:", session_attr["conversation_history"])
        
        print(f'sppech text:{response}')
        
        # memoryオブジェクトから会話の履歴を取得して、変数historyに代入
        history = memory.chat_memory

        return (
            handler_input.response_builder
            .speak(response)
            .ask("まだなんか聞きたいわけ?")
            .response
        )

class ChatGPTIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("ChatGPTIntent")(handler_input)

    def handle(self, handler_input):
        slots = handler_input.request_envelope.request.intent.slots
        question = slots["question"].value
        
        llm = ChatOpenAI(temperature=1.0, openai_api_key=os.environ['API_Key'], model_name="gpt-4-1106-preview")
        
        system_prompt = SystemMessagePromptTemplate.from_template(
            config.asuka_alexa_prompt
            )
            
        memory = ConversationSummaryBufferMemory(
            llm=llm, max_token_limit=100, return_messages=True
        )
        
        prompts = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
        chain = ConversationChain(
            llm=llm, prompt=prompts, memory=memory, verbose=True
        )
        
        
        response = chain.run(question)
        print(f'langcainの結果:{response}')
        memory.load_memory_variables({})
        return (
            handler_input.response_builder
            .speak(response)
            .ask("まだなんか聞きたいわけ?")
            .response
        )

sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(ChatGPTIntentHandler())
#sb.add_request_handler(LCChatGPTIntentHandler())
#
#sb.add_request_handler(HelloWorldIntentHandler())
#
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
# make sure IntentReflectorHandler is last so it doesn't override your custom intent handlers
sb.add_request_handler(IntentReflectorHandler())

sb.add_global_request_interceptor(LocalizationInterceptor())

sb.add_exception_handler(CatchAllExceptionHandler())

handler = sb.lambda_handler()