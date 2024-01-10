from loguru import logger
import json
import requests
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from dotenv import load_dotenv
import os
from app.utils.openai_helpers import OpenAIHelper

load_dotenv()


class WhatsAppMessages:
    def __init__(self) -> None:
        self.WHATSAPP_HEADERS = {
            "Content-type": "application/json",
            "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
        }
        self.WHATSAPP_URL = f"https://graph.facebook.com/{os.getenv('VERSION')}/{os.getenv('PHONE_NUMBER_ID')}/messages"

    @staticmethod
    def _get_text_message_format(recipient_number: str, text: str) -> json:
        """Generate the text message format."""
        return json.dumps(
            {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": recipient_number,
                "type": "text",
                "text": {"preview_url": False, "body": text},
            }
        )

    async def _send_message(self, recipient_number: str, text: str) -> JSONResponse:
        """Send a message using WhatsApp API."""
        try:
            data = self._get_text_message_format(recipient_number, text)
            response = requests.post(
                url=f"https://graph.facebook.com/{os.getenv('VERSION')}/{os.getenv('PHONE_NUMBER_ID')}/messages",
                data=data,
                headers={
                    "Content-type": "application/json",
                    "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
                },
                timeout=10,
            )
        except requests.Timeout:
            logger.exception("Timeout occurred while sending the message")
            return JSONResponse(
                content={"status": "error", "message": "Request timed out"},
                status_code=408,
            )
        except requests.RequestException as exc:
            logger.exception(f"Failed to send the message: {exc}")
            return JSONResponse(
                content={"status": "error", "message": "Failed to send the message"},
                status_code=500,
            )
        except Exception as exc:
            logger.exception(f"Unexpected error while sending the message: {exc}")
            return JSONResponse(
                content={"status": "error", "message": "Failed to send the message"},
                status_code=500,
            )

    async def error_message(self, recipient_number: str, text: str) -> JSONResponse:
        """Send an error message using WhatsApp API."""
        return await self._send_message(recipient_number, text)

    async def text_message(self, recipient_number: str, text: str) -> JSONResponse:
        """Send a text message using WhatsApp API."""
        return await self._send_message(recipient_number, text)
