import json
import openai
import os
import requests

from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.utils import is_request_type, is_intent_name
from ask_sdk_model import Response
from ask_sdk_model.ui import SimpleCard

sb = SkillBuilder()

# config
openai.api_key = os.environ["OPENAI_API_KEY"]
model = "gpt-3.5-turbo"
temperature = 0.9
max_tokens = 100
prompt_slot = "prompt"
system_prompt = """
You are a conversational AI. You should respond casually as though you are talking to a friend.
Do not give overly verbose answers with long rationalizations for your thoughts, you can have normal length responses.
If you would like and it's appropriate for the situation, you can even ask follow-up questions at the end of your response, but don't do this all the time.
"""

class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speech_text = "Welcome to Chatter Bot, let's chat!"

        handler_input.response_builder.speak(speech_text).set_card(
            SimpleCard("Chatter Bot", speech_text)).set_should_end_session(False)
        return handler_input.response_builder.response

class InvokeChatGPTIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return is_intent_name("InvokeChatGPTIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        slots = handler_input.request_envelope.request.intent.slots
        if prompt_slot not in slots:
            speech_text = "Sorry, I'm not sure I understood that."
            handler_input.response_builder.speak(speech_text).set_card(
                SimpleCard("Chatter Bot", speech_text)).set_should_end_session(True)
            return handler_input.response_builder.response
        
        phrase = slots[prompt_slot].value
        message_list = []
        
        # system message
        system_message = {
            "role": "system",
            "content": system_prompt
        }
        message_list.append(system_message)

        # user message
        user_message = {
            "role": "user",
            "content": phrase
        }
        message_list.append(user_message)

        # make request
        response_message = openai.ChatCompletion.create(
            model=model,
            messages=message_list,
            temperature=temperature,
            max_tokens=max_tokens
        )
        speech_text = response_message.choices[0].message.content

        handler_input.response_builder.speak(speech_text).set_card(
            SimpleCard("Chatter Bot", speech_text)).set_should_end_session(True)
        return handler_input.response_builder.response

class HelpIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speech_text = "You can actually ask me anything!"

        handler_input.response_builder.speak(speech_text).ask(speech_text).set_card(
            SimpleCard("Chatter Bot", speech_text))
        return handler_input.response_builder.response

class CancelAndStopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return is_intent_name("AMAZON.CancelIntent")(handler_input) \
            or is_intent_name("AMAZON.StopIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speech_text = "Goodbye!"

        handler_input.response_builder.speak(speech_text).set_card(
            SimpleCard("Chatter Bot", speech_text)).set_should_end_session(True)
        return handler_input.response_builder.response

class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        # any cleanup logic goes here

        return handler_input.response_builder.response

class AllExceptionHandler(AbstractExceptionHandler):

    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> Response
        # Log the exception in CloudWatch Logs
        print(exception)

        speech = "Sorry, I didn't get it. Can you please say it again!!"
        handler_input.response_builder.speak(speech).ask(speech)
        return handler_input.response_builder.response

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(InvokeChatGPTIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelAndStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())

sb.add_exception_handler(AllExceptionHandler())

lambda_handler = sb.lambda_handler()