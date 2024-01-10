import logging
import json

from typing import Any, Optional, Type

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool

from utilities.email.send import send_email

logger = logging.getLogger(__name__)

class SendEmailInput(BaseModel):
    to_email: str = Field(description="Email recipient's email address")
    subject: str = Field(description="Email subject")
    text_part: str = Field(description="Email text content")

class Email_SendEmail(BaseTool):
    name: str = "send_email"
    description: str = "Send text email to recipient"
    args_schema: Type[BaseModel] = SendEmailInput

    def _run(self,
        to_email: str,
        subject: str,
        text_part: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        logger.info(to_email)
        logger.info(subject)
        logger.info(text_part)
        try:
            send_email(to_email, subject, text_part)
            return json.dumps({
                "message": "send email OK"
            }, indent=4)
        except ValueError as e:
            return str(e)

