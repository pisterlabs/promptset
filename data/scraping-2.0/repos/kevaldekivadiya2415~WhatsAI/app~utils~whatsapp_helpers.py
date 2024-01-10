from loguru import logger
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from app.utils.openai_helpers import OpenAIHelper
from app.utils.templates.whatsapp_default_messages import (
    DEFAULT_ERROR_MESSAGE,
    MEDIA_NOT_SUPPORT_MESSAGE,
)
from app.utils.send_whatsapp_message import WhatsAppMessages

# Load environment variables
load_dotenv()

OPENAI_HELPER = OpenAIHelper()
WHATSAPP_MESSAGES = WhatsAppMessages()


class WhatsAppHandler:
    def __init__(self):
        pass

    @staticmethod
    def is_valid_whatsapp_message(message_body: dict) -> bool:
        """Check if the incoming webhook event has a valid WhatsApp message structure."""
        return (
            message_body.get("object")
            and message_body.get("entry")
            and message_body["entry"][0].get("changes")
            and message_body["entry"][0]["changes"][0].get("value")
            and message_body["entry"][0]["changes"][0]["value"].get("messages")
            and message_body["entry"][0]["changes"][0]["value"]["messages"][0]
        )

    async def generate_response(self, message: dict) -> dict:
        """Generate a response based on the type of WhatsApp message."""
        try:
            if message["type"] == "text":
                generated_text = await OPENAI_HELPER.text_response_generation(
                    text=message["text"]["body"]
                )
                return {"status": "success", "message": generated_text}
            else:
                return {"status": "fail", "message": MEDIA_NOT_SUPPORT_MESSAGE}
        except Exception as exc:
            logger.exception(exc)
            return {"status": "error", "message": DEFAULT_ERROR_MESSAGE}

    async def process_whatsapp_message(
        self, messages: list, recipient_number: str
    ) -> JSONResponse:
        """Process incoming WhatsApp messages and send a response."""
        try:
            # Generate response from GPT model
            response = await self.generate_response(message=messages[0])

            # Send generated text to the recipient's WhatsApp
            return await WHATSAPP_MESSAGES.text_message(
                recipient_number=recipient_number, text=response["message"]
            )
        except Exception as exc:
            logger.exception(f"Error processing WhatsApp message: {exc}")

            # If any exception occurred, send an error message
            return await WHATSAPP_MESSAGES.error_message(
                recipient_number=recipient_number, text=DEFAULT_ERROR_MESSAGE
            )
