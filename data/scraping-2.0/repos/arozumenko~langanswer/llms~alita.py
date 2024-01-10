# Copyright (c) 2023 Artem Rozumenko
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
import json

from typing import Any, List, Optional
from langchain.llms.base import LLM
from langchain_core.pydantic_v1 import Field


class AlitaLLM(LLM):
    
    model_name: str = "gpt-4"
    deployment: str = "https://eye.projectalita.ai"
    api_token: Optional[str] = None
    project_id: Optional[int] = None
    integration_id: Optional[str] = None
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.9
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 20
    stream_response: Optional[bool] = Field(default=False, alias="stream")

    def __init__(self, deployment: str, api_token: str, project_id: int, 
                 integration_id: str, model_name: Optional[str], 
                 max_tokens: Optional[int], temperature: Optional[float], 
                 top_p: Optional[float], top_k: Optional[int]):
        self.deployment = deployment
        self.api_token = api_token
        self.project_id = project_id
        self.integration_id = integration_id
        self.model_name = model_name if model_name else self.model_name
        self.max_tokens = max_tokens if max_tokens else self.max_tokens
        self.temperature = temperature if temperature else self.temperature
        self.top_p = top_p if top_p else self.top_p
        self.top_k = top_k if top_k else self.top_k

    @property
    def _llm_type(self) -> str:
        """
        This should return the type of the LLM.
        """
        return self.model_name
    
    
    def _call(self, prompt:str, **kwargs: Any,):
        """
        This is the main method that will be called when we run our LLM.
        """
        predict_url = f'{self.deployment}/prompt_lib/predict/prompt_lib/{self.project_id}'
        prompt_data = {
            "type": "freeform",
            "integration_id": self.integration_id,
            "project_id": self.project_id,
            "model_settings": self._get_model_default_parameters,
            "messages": [{
                "role": "user",
                "content": prompt
            }]
        }
        headers = {
            "Content-Type": "application/json",
            # Assuming you have the following variables defined: authType, authToken
            "Authorization": f'Bearer {self.api_token}',
        }
        response = requests.post(predict_url, headers=headers, data=json.dumps(prompt_data))
        response_data = response.json()
        return response_data['data']


    @property
    def _get_model_default_parameters(self):
        return  {
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "stream": self.stream_response,
                "model": {
                    "name": self.model_name,
                    "integration_uid": self.integration_id,
                }
            }

    @property
    def _identifying_params(self) -> dict:
        """
        It should return a dict that provides the information of all the parameters 
        that are used in the LLM. This is useful when we print our llm, it will give use the 
        information of all the parameters.
        """
        return {
            "deployment": self.deployment,
            "api_token": self.api_token,
            "project_id": self.project_id,
            "integration_id": self.integration_id,
            "model_settings": self._get_model_default_parameters,
        }
