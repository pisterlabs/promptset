import json
import re
import time
from typing import List, Optional

import openai

from openplugin.interfaces.models import (
    LLM,
    Config,
    Message,
    MessageType,
    Plugin,
    PluginDetected,
    SelectedPluginsResponse,
)
from openplugin.interfaces.plugin_selector import PluginSelector

plugin_prompt = """
{name_for_model}: Call this tool to get the OpenAPI spec (and usage guide) for interacting with the {name_for_model} API. 
You should only call this ONCE! 

What is the {name_for_model} API useful for? {description_for_model}.
"""  # noqa: E501

plugin_identify_prompt = """
Answer the following questions as best you can. You have access to the following tools:
{all_plugin_info_prompt}
Use the following format. Only reply with the action you want to take.:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of {all_plugin_names}, None if you don't want to use any tool
Begin!
Question: {prompt}
"""  # noqa: E501

plugin_operation_prompt = """
// You are an AI assistant.
// Here is a tool you can use, named {name_for_model}. The description for this plugin is: {description_for_model}.
// The Plugin rules:
// 1. Assistant ALWAYS asks user's input for ONLY the MANDATORY parameters BEFORE calling the API.
// 2. Assistant pays attention to instructions given below.
// 3. Create an HTTPS API url that represents this query.
// 4. Use this format: <HTTP VERB> <URL>
//   - An example: GET https://api.example.com/v1/products
// 5. Remove any starting periods and new lines.
// 6. Do not structure as a sentence.
// 7. Never use https://api.example.com/ in the API.

{pre_prompt}

The openapi spec file = {openapi_spec}
The instructions are: {prompt}
"""  # noqa: E501


# Function to extract URLs from text using regular expressions
def _extract_urls(text):
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    urls = re.findall(url_pattern, text)
    return urls


class ImpromptPluginSelector(PluginSelector):
    def __init__(
        self,
        plugins: List[Plugin],
        config: Optional[Config],
        llm: Optional[LLM],
    ):
        super().__init__(plugins, config, llm)
        self.total_tokens_used = 0
        if config and config.openai_api_key:
            openai.api_key = config.openai_api_key
        else:
            raise ValueError("OpenAI API Key is not configured")

    def run(self, messages: List[Message]) -> SelectedPluginsResponse:
        start_test_case_time = time.time()
        plugin_operations = self.get_detected_plugin_with_operations(messages)
        response = SelectedPluginsResponse(
            run_completed=True,
            final_text_response=None,
            detected_plugin_operations=plugin_operations,
            response_time=round(time.time() - start_test_case_time, 2),
            tokens_used=self.total_tokens_used,
            llm_api_cost=0,
        )
        return response

    def get_detected_plugin_with_operations(
        self, messages: List[Message]
    ) -> List[PluginDetected]:
        prompt = ""
        for message in messages:
            prompt += f"{message.message_type}: {message.content}\n"

        plugin_info_prompts = []
        plugin_names = []

        for plugin in self.plugins:
            if plugin.name:
                plugin_names.append(plugin.name)
            plugin_info_prompt = plugin_prompt.format(
                name_for_model=plugin.name, description_for_model=plugin.description
            )
            plugin_info_prompts.append(plugin_info_prompt)
        plugin_detection_prompt = plugin_identify_prompt.format(
            all_plugin_info_prompt="".join(plugin_info_prompts),
            all_plugin_names=", ".join(plugin_names),
            prompt=prompt,
        )

        response = self.run_llm_prompt(plugin_detection_prompt)
        found_plugins = []
        for line in response.get("response").splitlines():
            if line.strip().startswith("Action"):
                for val in line.split("Action:"):
                    if len(val.strip()) > 0:
                        if "-" in val:
                            val = val.split("-")[0].strip()
                        if "," in val:
                            for v in val.split(","):
                                found_plugins.append(
                                    self.get_plugin_by_name(v.strip())
                                )
                        else:
                            found_plugins.append(
                                self.get_plugin_by_name(val.strip())
                            )

        detected_plugins = []
        for plugin in found_plugins:
            if plugin is None:
                continue
            api_called = None
            # TODO Find a better way to find the API called
            openapi_spec_json = plugin.get_openapi_doc_json()
            formatted_plugin_operation_prompt = plugin_operation_prompt.format(
                name_for_model=plugin.name,
                description_for_model=plugin.description,
                pre_prompt=plugin.get_plugin_pre_prompts(),
                openapi_spec=json.dumps(openapi_spec_json),
                prompt=prompt,
            )
            response = self.run_llm_prompt(formatted_plugin_operation_prompt)
            method = "get"
            if "post" in response.get("response").lower():
                method = "post"
            elif "put" in response.get("response").lower():
                method = "put"
            elif "delete" in response.get("response").lower():
                method = "delete"
            urls = _extract_urls(response.get("response"))
            for url in urls:
                formatted_url = url.split("?")[0].strip()
                if plugin.api_endpoints and formatted_url in plugin.api_endpoints:
                    api_called = formatted_url
                    break
            detected_plugins.append(
                PluginDetected(
                    plugin=plugin,
                    api_called=api_called,
                    method=method,
                )
            )
        return detected_plugins

    def run_llm_prompt(self, prompt):
        if self.llm.provider.lower() == "openai":
            msgs = [{"role": "user", "content": prompt}]
            return self.openai_chat(msgs)
        raise ValueError(f"LLM provider {self.llm.provider} not supported")

    def run_llm(self, messages: List[Message]):
        if self.llm is None:
            raise ValueError("LLM is not configured")
        if self.llm.provider.lower() == "openai":
            msgs = []
            for message in messages:
                if message.message_type == MessageType.HumanMessage:
                    role = "user"
                elif message.message_type == MessageType.AIMessage:
                    role = "assistant"
                elif message.message_type == MessageType.SystemMessage:
                    role = "system"
                msgs.append({"role": role, "content": message.content})
            return self.openai_chat(messages)
        raise ValueError(f"LLM provider {self.llm.provider} not supported")

    def openai_chat(self, messages):
        response = openai.ChatCompletion.create(
            model=self.llm.model_name,
            messages=messages,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
            top_p=self.llm.top_p,
            frequency_penalty=self.llm.frequency_penalty,
            presence_penalty=self.llm.presence_penalty,
        )
        self.add_to_tokens(response.get("usage").get("total_tokens"))
        return {
            "response": response.get("choices")[0].get("message").get("content"),
            "usage": response.get("usage"),
        }

    def openai_completion(self, prompt):
        response = openai.Completion.create(
            model=self.llm.model_name,
            prompt=prompt,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
            top_p=self.llm.top_p,
            frequency_penalty=self.llm.frequency_penalty,
            presence_penalty=self.llm.presence_penalty,
        )
        self.add_to_tokens(response.get("usage").get("total_tokens"))
        return {
            "response": response.get("choices")[0].get("text"),
            "usage": response.get("usage"),
        }

    def add_to_tokens(self, tokens):
        if tokens:
            self.total_tokens_used += tokens

    @classmethod
    def get_pipeline_name(cls) -> str:
        return "imprompt basic"
