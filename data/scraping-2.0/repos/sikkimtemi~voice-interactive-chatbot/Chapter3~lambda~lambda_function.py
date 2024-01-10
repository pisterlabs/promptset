import logging
import ask_sdk_core.utils as ask_utils
import openai
import time

from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components.request_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components.exception_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model.dialog.elicit_slot_directive import ElicitSlotDirective
from ask_sdk_model.intent import Intent
from ask_sdk_model.intent_confirmation_status import IntentConfirmationStatus
from ask_sdk_model.slot import Slot
from ask_sdk_model.slot_confirmation_status import SlotConfirmationStatus

from ask_sdk_model.response import Response

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "私はAIチャットボットです。何でも聞いてください。"

        directive = ElicitSlotDirective(
            slot_to_elicit="user_message",
            updated_intent = Intent(
                name = "ChatBotIntent",
                confirmation_status = IntentConfirmationStatus.NONE,
                slots ={
                    "user_message": Slot(name= "user_message", value = "", confirmation_status = SlotConfirmationStatus.NONE)
                }
            )
        )

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("なにか話しかけてください。")
                .add_directive(directive)
                .response
        )


class ChatBotIntentHandler(AbstractRequestHandler):
    """Handler for ChatBot Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("ChatBotIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response

        # 開始時刻
        start_time = time.time()

        # OpenAIのAPIキーを設定
        openai.api_key = "your-api-key"

        # プロンプトの準備
        template = """あなたは音声対話型チャットボットです。以下の制約にしたがって回答してください。
        制約：
        - ユーザーのメッセージに句読点を補ってから回答します
        - 140文字以内を目安に簡潔な短い文章で話します
        - 質問の答えがわからない場合は「わかりません」と答えます"""

        # メッセージの初期化
        messages = [
            {
                "role": "system",
                "content": template
            }
        ]

        # セッションからメッセージを取り出す
        if "MESSAGES" in handler_input.attributes_manager.session_attributes:
            messages = handler_input.attributes_manager.session_attributes["MESSAGES"]

        user_input = ask_utils.get_slot_value(handler_input=handler_input, slot_name="user_message")

        # ユーザーのメッセージを追加
        messages.append({
            "role": "user",
            "content": user_input if isinstance(user_input, str) else "こんにちは"
        })

        try:
            # Streamingを有効にしてOpenAIのAPIを呼び出す
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                stream=True
            )
            message = ""
            for chunk in response:
                elapsed_time = time.time() - start_time
                finish_reason = chunk['choices'][0]['finish_reason']
                if finish_reason != 'stop':
                    message += chunk['choices'][0]['delta']['content']
                if elapsed_time > 7.9:
                    message += "。すみません、タイムアウトしました。"
                    break
            speak_output = message
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            speak_output = "すみません、エラーが発生しました。しばらく時間をおいてからもう一度お試しください。"

        # ChatGPTの回答をメッセージに追加
        messages.append({
            "role": "assistant",
            "content": speak_output
        })

        # セッションにメッセージを保存
        handler_input.attributes_manager.session_attributes["MESSAGES"] = messages

        directive = ElicitSlotDirective(
            slot_to_elicit="user_message",
            updated_intent = Intent(
                name = "ChatBotIntent",
                confirmation_status = IntentConfirmationStatus.NONE,
                slots ={
                    "user_message": Slot(name= "user_message", value = "", confirmation_status = SlotConfirmationStatus.NONE)
                }
            )
        )

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("なにか話しかけてください。")
                .add_directive(directive)
                .response
        )


class HelpIntentHandler(AbstractRequestHandler):
    """Handler for Help Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.predicate.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "自由に話しかけてみてください。"

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
        return (ask_utils.predicate.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.predicate.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "さようなら。"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .response
        )

class FallbackIntentHandler(AbstractRequestHandler):
    """Single handler for Fallback Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.predicate.is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        logger.info("In FallbackIntentHandler")
        speech = "すみません、よくわかりません。"
        reprompt = "お助けできることは何かありますか？"

        return handler_input.response_builder.speak(speech).ask(reprompt).response

class SessionEndedRequestHandler(AbstractRequestHandler):
    """Handler for Session End."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.predicate.is_request_type("SessionEndedRequest")(handler_input)

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
        return ask_utils.predicate.is_request_type("IntentRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        intent_name = ask_utils.request_util.get_intent_name(handler_input)
        speak_output = "あなたが呼び出したインテントはこちらです。{}".format(intent_name)

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

        speak_output = "申し訳ありません。もう一度お試しください。"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

# The SkillBuilder object acts as the entry point for your skill, routing all request and response
# payloads to the handlers above. Make sure any new handlers or interceptors you've
# defined are included below. The order matters - they're processed top to bottom.


sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(ChatBotIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_request_handler(IntentReflectorHandler()) # make sure IntentReflectorHandler is last so it doesn't override your custom intent handlers

sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
