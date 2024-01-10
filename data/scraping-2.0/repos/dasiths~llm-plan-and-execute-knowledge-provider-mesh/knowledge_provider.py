from pydantic import BaseModel, Field
from typing import Optional, Callable, Type
from langchain.tools import BaseTool
import requests

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class KnowledgeProviderToolInput(BaseModel):
    request_payload: dict = Field()
    metadata: Optional[dict] = Field(default=None)

class KnowledgeProviderServiceInput():
    request_id: str
    payload: dict

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "payload": self.payload
        }

class KnowledgeProviderServiceOutput():
    output: str

    def __new__(cls, dict: dict):
        instance = super().__new__(cls)
        instance.output = dict["output"]
        return instance

class KnowledgeProviderTool(BaseTool):
    args_schema: Type[BaseModel] = KnowledgeProviderToolInput

    def __int__(self, name, description, **data: any):
        self.name = name
        self.description = description
        super(KnowledgeProviderTool, self).__init__(**data)

    def _run(
        self, request_payload: dict, metadata: dict, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        print("input is ", request_payload)
        print("meta data is ", metadata)

        if "call_handler" in self.metadata:
            if callable(self.metadata["call_handler"]):
                return self.metadata["call_handler"](request_payload, metadata)

        raise Exception("no handler provided")

    async def _arun(
        self, request_payload: dict, metadata: dict, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("tool does not support async")

def create_tool(name, description, call_handler: Callable[[dict, dict], str]=None, **data: any):
    tool = KnowledgeProviderTool(name=name, description=description, **data)
    if call_handler:
        tool.metadata = {}
        tool.metadata["call_handler"] = call_handler
    return tool

class KnowledgeProvider():
    botTool: KnowledgeProviderTool
    url: str

    def __new__(cls, name, description, url, **data: any):
        instance = super().__new__(cls)
        instance.botTool = create_tool(name, description, instance.call_service, **data)
        instance.url = url
        return instance

    def get_tool(self) -> KnowledgeProviderTool:
        return self.botTool

    def call_service(self, input: dict, metadata: dict) -> str:
        # Making the POST request
        request_obj = KnowledgeProviderServiceInput()
        request_obj.request_id = "123"
        request_obj.payload = input

        response = requests.post(self.url, json=request_obj.to_dict())

        # Parsing and printing the response content
        if response.ok:
            response_data = KnowledgeProviderServiceOutput(response.json())
            print("Response from knowledge provider: ", response_data.output)
            return response_data.output
        else:
            raise Exception("Request failed with status code:", response.status_code)
