from flask import Blueprint, request, make_response, Response
import logging

from ..config import OpenAIContent, WhatsAppData, FLASK_ENV
from .services.whatsapp import (
    authenticate_token,
    get_whatsapp_messages,
    get_user_id_from_whatsapp_message,
    get_content_from_whatsapp_message,
    send_whatsapp_message,
)
from .services.conversation import (
    get_or_create_conversation,
    add_message_to_conversation,
)
from .services.openai import get_assistant_content

logger = logging.getLogger(__name__)

webhook_route_blueprint = Blueprint("webhook", __name__, url_prefix="/webhook")


@webhook_route_blueprint.route("/", methods=["GET"])
def webhook_get() -> Response:
    logger.info("Received GET request to webhook")

    hub_mode = request.args.get("hub.mode")
    verify_token = request.args.get("hub.verify_token")

    if authenticate_token(hub_mode, verify_token):
        body = request.args.get("hub.challenge") or "OK"
        logger.info("Verification successful")
        return make_response(body, 200)
    else:
        logger.warning("Invalid verification token")
        return make_response("Invalid verification token", 401)


@webhook_route_blueprint.route("/", methods=["POST"])
def webhook_post() -> Response:
    logger.info("Received POST request to webhook")
    if request.json is None:
        logger.warning("Received request with no JSON data.")
        return make_response("No data", 400)

    try:
        data: WhatsAppData = request.json

        whatsapp_messages = get_whatsapp_messages(data)

        for whatsapp_message in whatsapp_messages:
            user_id = get_user_id_from_whatsapp_message(whatsapp_message)
            user_content = get_content_from_whatsapp_message(whatsapp_message)

            conversation = get_or_create_conversation(user_id)

            add_message_to_conversation(
                conversation,
                role="user",
                content=user_content,
            )

            assistant_content: OpenAIContent = get_assistant_content(
                conversation.limited_messages
            )
            # TODO: Fake message for testing purposes. Remove.
            # assistant_content: OpenAIContent = "Não se preocupe, estou aqui para te ajudar a entender e gerenciar suas finanças pessoais de forma simples e eficiente. Posso te orientar desde a criação de um orçamento até a organização de investimentos para o futuro."

            add_message_to_conversation(
                conversation,
                role="assistant",
                content=assistant_content,
            )

            send_whatsapp_message(user_id, assistant_content)
    except Exception as e:
        logger.error(
            f"An error occurred: {str(e)}. Cause: {str(e.__cause__)}",
            exc_info=(FLASK_ENV == "development"),
        )
        return make_response(str(e), 500)

    return make_response("OK", 200)
