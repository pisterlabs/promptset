import logging
import requests

from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_core.utils import is_request_type, is_intent_name
from ask_sdk_model import Intent, Slot
from ask_sdk_model.dialog import ElicitSlotDirective
from ask_sdk_model.interfaces.alexa.presentation.apl import RenderDocumentDirective, UpdateIndexListDataDirective
from constants import prompts
from constants.apl_constants import BubbleChatAPL
from constants.intent_constants import QUESTION_INTENT_NAME, RequestType, QUESTION_INTENT_QUESTION_SLOT_NAME, \
    QUESTION_INTENT_MAX_FREE_INTERACTION_COUNT
from constants.openai_constants import OpenAIChatRequest
from constants.prompts import SUBSCRIPTION_UPSELL
from constants.secret_manager_constants import OPENAI_API_KEY_SECRET_MANAGER_KEY, OPENAI_API_KEY_SECRET_MANAGER_INDEX
from handlers.buy_subs_intent_handler import BuySubsIntentHandler
from handlers.cancel_subs_handler import CancelSubsIntentHandler
from handlers.help_intent_handler import HelpIntentHandler
from handlers.model_intent import ModelIntentHandler
from services.ddb_gateway import DynamoDBGateway
from services.openai_gateway import OpenAIGateway
from services.secret_manager_gateway import SecretManagerGateway
from utils.intent_dispatch_utils import is_clear_context_request, is_stop_session_request, is_buy_subs_request, \
    is_cancel_subs_request, supports_apl, is_help_request, is_update_model_request
from utils.isp_utils import is_entitled

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QuestionIntentHandler(AbstractRequestHandler):
    """Handler class for GPT_QuestionIntent"""

    def __init__(self):
        self.api_key = ""
        self.context = []
        self.data = None
        self.interaction_count = 0
        self.redirected_search_query = ""
        self.search_query = ""
        self.gpt_response = ""
        self.finish_reason = ""
        self.gpt_image_response = ""
        self.MAX_CHAT_CONTEXT = 6
        self.openai_gateway = OpenAIGateway()
        self.secret_manager_gateway = SecretManagerGateway()
        self.ddb_gateway = DynamoDBGateway()

    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return (is_request_type(RequestType.INTENT_REQUEST.value)(handler_input) and
                is_intent_name(QUESTION_INTENT_NAME)(handler_input))

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        try:
            logger.info("QuestionIntentHandler Handling Request")
            # Get Localization Data
            self.data = handler_input.attributes_manager.request_attributes["_"]

            # Check if user want to special intent
            if is_help_request(handler_input):
                help_handler = HelpIntentHandler()
                return help_handler.handle(handler_input)

            if is_stop_session_request(handler_input):
                speech = self.data[prompts.STOP_MESSAGE]
                handler_input.response_builder.speak(speech)
                return handler_input.response_builder \
                    .set_should_end_session(should_end_session=True) \
                    .response

            if is_clear_context_request(handler_input):
                # reset chat_context session attributes
                handler_input.attributes_manager.session_attributes["chat_context"] = []
                speech = self.data[prompts.CONTEXT_CLEAR_RESPONSE]
                handler_input.response_builder.speak(speech)
                handler_input.response_builder \
                    .add_directive(ElicitSlotDirective(
                        updated_intent=Intent(
                            name=QUESTION_INTENT_NAME,
                            slots={
                                "question": Slot(
                                    name=QUESTION_INTENT_QUESTION_SLOT_NAME,
                                    value=("{" + QUESTION_INTENT_QUESTION_SLOT_NAME + "}")
                                )}
                        ),
                        slot_to_elicit=QUESTION_INTENT_QUESTION_SLOT_NAME))
                return handler_input.response_builder.set_should_end_session(should_end_session=False).response

            if is_update_model_request(handler_input):
                model_intent_handler = ModelIntentHandler()
                slots = handler_input.request_envelope.request.intent.slots
                utterance_text = slots[QUESTION_INTENT_QUESTION_SLOT_NAME].value
                return model_intent_handler.handle_text_request(handler_input, utterance_text)

            if is_buy_subs_request(handler_input):
                buy_subs_handler = BuySubsIntentHandler()
                slots = handler_input.request_envelope.request.intent.slots
                utterance_text = slots[QUESTION_INTENT_QUESTION_SLOT_NAME].value
                return buy_subs_handler.handle_text_request(handler_input, utterance_text)

            if is_cancel_subs_request(handler_input):
                cancel_subs_handler = CancelSubsIntentHandler()
                slots = handler_input.request_envelope.request.intent.slots
                utterance_text = slots[QUESTION_INTENT_QUESTION_SLOT_NAME].value
                return cancel_subs_handler.handle_text_request(handler_input, utterance_text)

            # Fetch chat context from skill session
            self.process_chat_context(handler_input)

            # Verify membership and interaction count
            entiltement = is_entitled(handler_input)
            if (self.interaction_count > QUESTION_INTENT_MAX_FREE_INTERACTION_COUNT) \
                    and (not entiltement):
                return handler_input.response_builder \
                    .speak(self.data[SUBSCRIPTION_UPSELL]) \
                    .add_directive(ElicitSlotDirective(
                        updated_intent=Intent(
                            name=QUESTION_INTENT_NAME,
                            slots={
                                "question": Slot(
                                    name=QUESTION_INTENT_QUESTION_SLOT_NAME,
                                    value=("{" + QUESTION_INTENT_QUESTION_SLOT_NAME + "}")
                                )}
                        ),
                        slot_to_elicit=QUESTION_INTENT_QUESTION_SLOT_NAME)) \
                    .set_should_end_session(should_end_session=False) \
                    .response

            # Call OpenAI to get chat response
            self.get_api_key()
            self.get_gpt_response(handler_input)
            if self.finish_reason == "length":
                speech = self.data[prompts.QUESTION_UNFINISHED_RESPONSE].format(self.gpt_response)
            else:
                speech = self.data[prompts.QUESTION_RESPONSE].format(self.gpt_response)

            # Render APL Card
            self.launch_screen(handler_input)

            return handler_input.response_builder \
                .speak(speech) \
                .add_directive(ElicitSlotDirective(
                    updated_intent=Intent(
                        name=QUESTION_INTENT_NAME,
                        slots={
                            "question": Slot(
                                name=QUESTION_INTENT_QUESTION_SLOT_NAME,
                                value=("{" + QUESTION_INTENT_QUESTION_SLOT_NAME + "}")
                            )}
                    ),
                    slot_to_elicit=QUESTION_INTENT_QUESTION_SLOT_NAME)) \
                .set_should_end_session(should_end_session=False) \
                .response

        except Exception as exception:
            logger.exception("Failed to handle QuestionIntent")
            speech = self.data[prompts.QUESTION_INTENT_OPENAI_ERROR_MESSAGE]
            return handler_input.response_builder \
                .speak(speech) \
                .set_should_end_session(should_end_session=True) \
                .add_directive(ElicitSlotDirective(
                    updated_intent=Intent(
                        name=QUESTION_INTENT_NAME,
                        slots={
                            "question": Slot(
                                name=QUESTION_INTENT_QUESTION_SLOT_NAME,
                                value=("{" + QUESTION_INTENT_QUESTION_SLOT_NAME + "}")
                            )}
                    ),
                    slot_to_elicit=QUESTION_INTENT_QUESTION_SLOT_NAME)) \
                .response

    def handle_more_intent(self, handler_input):
        # type: (HandlerInput) -> Response
        self.redirected_search_query = "more"
        return self.handle(handler_input)

    def handle_customized_intent(self, handler_input, search_query):
        # type: (HandlerInput) -> Response
        self.redirected_search_query = search_query
        return self.handle(handler_input)

    def process_chat_context(self, handler_input):
        # type: (HandlerInput) -> None
        if self.redirected_search_query:
            self.search_query = self.redirected_search_query
        else:
            slots = handler_input.request_envelope.request.intent.slots
            self.search_query = slots[QUESTION_INTENT_QUESTION_SLOT_NAME].value

        chat_context = []
        self.interaction_count = 0
        try:
            chat_context = handler_input.attributes_manager.session_attributes["chat_context"]
            self.interaction_count = handler_input.attributes_manager.session_attributes["interaction_count"]
        except:
            logger.exception("Received null chat_context or interaction_count")

        if not chat_context:
            handler_input.attributes_manager.session_attributes["chat_context"] = []
            chat_context = []

        chat_context.append({"role": "user", "content": self.search_query})
        if len(chat_context) > self.MAX_CHAT_CONTEXT:
            chat_context = chat_context[2:]
        self.interaction_count += 1
        customer_id = handler_input.request_envelope.session.user.user_id
        self.ddb_gateway.update_interaction_count(customer_id, self.interaction_count)

        self.context = chat_context
        handler_input.attributes_manager.session_attributes["chat_context"] = chat_context
        handler_input.attributes_manager.session_attributes["interaction_count"] = self.interaction_count

    def get_api_key(self) -> str:
        secret_value = self.secret_manager_gateway.get_secret_value(OPENAI_API_KEY_SECRET_MANAGER_KEY)
        api_key = secret_value[OPENAI_API_KEY_SECRET_MANAGER_INDEX]
        self.api_key = api_key
        return api_key

    def get_gpt_response(self, handler_input):
        try:
            model_setting = handler_input.attributes_manager.session_attributes["model_setting"]
            chat_request = OpenAIChatRequest(self.api_key, context=self.context, model=model_setting)
            raw_responses = self.openai_gateway.call([chat_request], handler_input)
            self.gpt_response = raw_responses[chat_request.get_response_key()]["prompt"]
            full_message = raw_responses[chat_request.get_response_key()]["full_message"]
            self.finish_reason = raw_responses[chat_request.get_response_key()]["finish_reason"]
            logger.info("OpenAI Chat response: " + full_message)

            # Store text response content
            self.context.append({"role": "assistant", "content": full_message})
        except requests.exceptions.Timeout as exception:
            logger.exception("OpenAI API call timeout")
            self.gpt_response = self.data[prompts.QUESTION_INTENT_OPENAI_TIMEOUT_MESSAGE]
            self.finish_reason = "timeout"
        except Exception as exception:
            logger.exception("Failed to call OpenAI")
            raise exception

    def launch_screen(self, handler_input):
        try:
            # Only add APL directive if User's device supports APL
            if not supports_apl(handler_input):
                return

            # Get Bubble Chat APL
            apl = BubbleChatAPL()

            # Format chat context to APL data
            apl_chat_context = []
            for chat in self.context:
                if chat["role"] == "assistant":
                    apl_chat_context.append({
                        "type": "SpeechBubble",
                        "message": chat["content"],
                        "sender": "alexa"
                    })
                elif chat["role"] == "user":
                    apl_chat_context.append({
                        "type": "SpeechBubble",
                        "message": chat["content"]
                    })

            # Build APL card
            apl.set_chat_context(apl_chat_context)
            handler_input.response_builder.add_directive(
                RenderDocumentDirective(
                    token=apl.get_document_token(),
                    document={
                        "type": "Link",
                        "src": f"doc://alexa/apl/documents/{apl.get_document_id()}"
                    },
                    datasources=apl.get_data_source()
                )
            )
        except Exception as exception:
            logger.exception("Failed to render BubbleChat APL")

