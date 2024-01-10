import json
from typing import Type

import requests
import streamlit as st
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.tools import BaseTool
from langchain.utilities.zapier import ZapierNLAWrapper
from pydantic import BaseModel, Field
from requests.auth import HTTPBasicAuth

# Step1で使うツール


class NoOpTool(BaseTool):
    name = "noop"
    description = "Don't call this tool"

    def _run(self, query):
        return "noop"


# Step4で使うツール


def load_zapier_tools_for_openai_functions_agent():
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(ZapierNLAWrapper())
    tools = toolkit.get_tools()

    # LangChainのZapier NLAのツールは、descriptionが長すぎるため、
    # OpenAI Functions Agentではエラーになります。
    # エラーを回避するため、descriptionをnameで上書きしています。
    for tool in tools:
        tool.description = tool.name

    return tools


# Step5で使うツール


class ToggleLightInput(BaseModel):
    on: bool = Field(description="Whether to turn the light on or off")


class ToggleStreamlitImageLightTool(BaseTool):
    name = "toggle-light"
    description = "toggle the light on or off"
    args_schema: Type[BaseModel] = ToggleLightInput

    def _run(self, on):
        st.session_state.is_light_on = on
        return json.dumps({"is_light_on": on})


class ToggleFanInput(BaseModel):
    on: bool = Field(description="Whether to turn the fan on or off")


class ToggleStreamlitImageFanTool(BaseTool):
    name = "toggle-fan"
    description = "toggle the fan on or off"
    args_schema: Type[BaseModel] = ToggleFanInput

    def _run(self, on):
        st.session_state.is_fan_on = on
        return json.dumps({"is_fan_on": on})


def load_streamlit_image_tools() -> list[BaseTool]:
    return [
        ToggleStreamlitImageLightTool(),
        ToggleStreamlitImageFanTool(),
    ]


# Step5で使うツール


class ToogleRemoteLightTool(BaseTool):
    name = "toggle-light"
    description = "toggle the light on or off"
    args_schema: Type[BaseModel] = ToggleLightInput

    host: str
    room_id: str
    basic_auth_username: str
    basic_auth_password: str

    def _run(self, on):
        url = f"http://{self.host}/rooms/{self.room_id}/update"
        auth = HTTPBasicAuth(self.basic_auth_username, self.basic_auth_password)
        req_body = {"is_light_on": on}
        res = requests.post(url, auth=auth, json=req_body)
        return res.text
