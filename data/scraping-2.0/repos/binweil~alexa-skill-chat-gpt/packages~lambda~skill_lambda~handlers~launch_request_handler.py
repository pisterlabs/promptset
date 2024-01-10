import logging
import os
import traceback

import boto3
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_core.utils import is_request_type, get_supported_interfaces
from ask_sdk_model import Intent, Slot
from ask_sdk_model.dialog import ElicitSlotDirective
from ask_sdk_model.interfaces.alexa.presentation.apl import RenderDocumentDirective

from constants import prompts
from constants.apl_constants import HomeScreenAPL
from constants.intent_constants import QUESTION_INTENT_NAME, QUESTION_INTENT_QUESTION_SLOT_NAME, RequestType
from constants.openai_constants import OpenAIConfig
from services.ddb_gateway import DynamoDBGateway
from utils.intent_dispatch_utils import supports_apl
from utils.isp_utils import is_entitled

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Built-in Intent Handlers
class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch and LaunchRequest Intent."""

    def __init__(self):
        self.ddb_client = boto3.client('dynamodb')
        self.table_name = os.getenv('DYNAMODB_TABLE_NAME')
        self.ddb_gateway = DynamoDBGateway()

    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return is_request_type(RequestType.LAUNCH_REQUEST.value)(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        logger.info("LaunchRequestHandler Handling Request")

        # Initialize session attributes
        entiltement = is_entitled(handler_input)
        if entiltement:
            handler_input.attributes_manager.session_attributes["model_setting"] = \
            OpenAIConfig.GPT_MODEL_4.value
        else:
            handler_input.attributes_manager.session_attributes["model_setting"] = \
            OpenAIConfig.GPT_MODEL_3_5.value
        handler_input.attributes_manager.session_attributes["chat_context"] = []
        handler_input.attributes_manager.session_attributes["interaction_count"] = 0
        customer_id = handler_input.request_envelope.session.user.user_id
        customer_info = self.ddb_gateway.get(customer_id)

        # Fetch & Update customer settings from DDB
        if customer_info is None:
            self.ddb_gateway.put(customer_id, 0, handler_input.attributes_manager.session_attributes["model_setting"])
        else:
            # Update to customer["interaction_count"] if interaction_count check is accumulative later on
            handler_input.attributes_manager.session_attributes["interaction_count"] = 0
            self.ddb_gateway.update_interaction_count(customer_id, 0)
            self.ddb_gateway.update_model_setting(customer_id, handler_input.attributes_manager.session_attributes["model_setting"])

        # get localization data
        data = handler_input.attributes_manager.request_attributes["_"]


        if entiltement:
            speech = data[prompts.LAUNCH_MESSAGE_ENTITLED]
        else:
            speech = data[prompts.LAUNCH_MESSAGE]

        try:
            self.launch_screen(handler_input)
        except Exception as exception:
            logger.exception("Failed to render LaunchRequest APL card")

        handler_input.response_builder.speak(speech)

        return handler_input.response_builder\
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
                slot_to_elicit=QUESTION_INTENT_QUESTION_SLOT_NAME))\
            .response


    def launch_screen(self, handler_input):
        # Only add APL directive if User's device supports APL
        apl = HomeScreenAPL()
        if supports_apl(handler_input):
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

