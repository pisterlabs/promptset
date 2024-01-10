from typing import Optional
import json

from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from pydantic import BaseModel, EmailStr
from googleapiclient.discovery import Resource

from jsonllm.client.langchain_client import JsonSchemaLLM
from homellm.google_services import send_email


class SendEmailParameters(BaseModel):
    subject: str
    body: str
    recipient: EmailStr


_prompt = """\
A description of an email:
{{description}}

The json object describing the email:

"""


def get_email_parameters(input: str, llm: JsonSchemaLLM) -> SendEmailParameters:
    prompt = _prompt.format(description=input)
    return SendEmailParameters(**json.loads(llm(prompt=prompt)))


class SendEmailTool(BaseTool):
    name = "send_email"
    description = (
        "Useful for sending emails, a subject, body, and recipient are required."
    )
    gmail_service: Resource
    llm: JsonSchemaLLM = None

    def __init__(self, *args, llm=None, **kwargs):
        llm = llm or JsonSchemaLLM(schema_restriction=SendEmailParameters.schema())
        super().__init__(*args, llm=llm, **kwargs)

    def _run(
        self,
        input: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        inputs = get_email_parameters(input, self.llm)
        send_email(
            **inputs.dict(),
            gmail_service=self.gmail_service,
        )

    async def _arun(
        self,
        subject: str,
        body: str,
        recipient: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
