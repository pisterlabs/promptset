import logging

from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_core.utils import is_request_type, is_intent_name
from ask_sdk_model import Intent, Slot
from ask_sdk_model.dialog import ElicitSlotDirective
from ask_sdk_runtime.dispatch_components import AbstractRequestHandler

from constants import prompts
from constants.intent_constants import MODEL_INTENT_NAME, MODEL_INTENT_SLOT_NAME, RequestType, QUESTION_INTENT_NAME, \
    QUESTION_INTENT_QUESTION_SLOT_NAME
from constants.openai_constants import OpenAIConfig
from services.ddb_gateway import DynamoDBGateway

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ModelIntentHandler(AbstractRequestHandler):
    """Handler class for GPT_ModelIntent"""

    def __init__(self):
        self.model_version = OpenAIConfig.GPT_MODEL_3_5
        self.data = None
        self.ddb_gateway = DynamoDBGateway()

    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return (is_request_type(RequestType.INTENT_REQUEST.value)(handler_input) and
                is_intent_name(MODEL_INTENT_NAME)(handler_input))

    def handle_text_request(self, handler_input, utterance_text):
        # type: (HandlerInput) -> Response
        self.data = handler_input.attributes_manager.request_attributes["_"]
        speech = self.data[prompts.MODEL_UPDATE_ERROR_MESSAGE]
        try:
            customer_id = handler_input.request_envelope.session.user.user_id
            interaction_count = handler_input.attributes_manager.session_attributes["interaction_count"]

            if ("3" in utterance_text) or ("three" in utterance_text):
                speech = self.data[prompts.MODEL_UPDATE_RESPONSE].format("GPT 3.5 Turbo")
                self.ddb_gateway.update_model_setting(customer_id, OpenAIConfig.GPT_MODEL_3_5.value)
                handler_input.attributes_manager.session_attributes["model_setting"] \
                    = OpenAIConfig.GPT_MODEL_3_5.value
            elif ("4" in utterance_text) or ("three" in utterance_text):
                speech = self.data[prompts.MODEL_UPDATE_RESPONSE].format("GPT 4")
                self.ddb_gateway.update_model_setting(customer_id, OpenAIConfig.GPT_MODEL_4.value)
                handler_input.attributes_manager.session_attributes["model_setting"] \
                    = OpenAIConfig.GPT_MODEL_4.value
            else:
                raise Exception("Invalid model selection")
            return handler_input.response_builder \
                .speak(speech) \
                .set_should_end_session(should_end_session=False) \
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
        except Exception as exception:
            logger.exception("Failed to process Model update")
            return handler_input.response_builder \
                .speak(speech) \
                .set_should_end_session(should_end_session=False) \
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

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        self.data = handler_input.attributes_manager.request_attributes["_"]
        slots = handler_input.request_envelope.request.intent.slots
        model = slots[MODEL_INTENT_SLOT_NAME].value
        return self.handle_text_request(handler_input, model)





