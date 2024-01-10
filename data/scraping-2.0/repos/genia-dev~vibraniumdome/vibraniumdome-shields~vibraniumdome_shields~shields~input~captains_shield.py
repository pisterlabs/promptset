import os
import logging
from typing import List
from uuid import UUID

import openai
from openai.error import APIConnectionError, APIError, Timeout, TryAgain
from pydantic import Json
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from vibraniumdome_shields.settings_loader import settings
from vibraniumdome_shields.shields.model import LLMInteraction, ShieldDeflectionResult, VibraniumShield
from vibraniumdome_shields.utils import safe_loads_dictionary_string


class CaptainsShieldDeflectionResult(ShieldDeflectionResult):
    llm_response: Json


class CaptainsShield(VibraniumShield):
    logger = logging.getLogger(__name__)
    _shield_name: str = "com.vibraniumdome.shield.input.captain"

    def __init__(self, openai_api_key):
        super().__init__(self._shield_name)
        if not openai_api_key:
            raise ValueError("LLMShield missed openai_api_key")
        openai.api_key = openai_api_key

    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((Timeout, TryAgain, APIError, APIConnectionError)),
    )
    def deflect(self, llm_interaction: LLMInteraction, shield_policy_config: dict, scan_id: UUID, policy: dict) -> List[ShieldDeflectionResult]:
        self.logger.info("performing scan, id='%s'", scan_id)
        llm_shield_prompt = settings[self._shield_name]["prompt"]
        llm_message = llm_interaction.get_all_user_messages_or_function_results()
        messages = [
            {"role": "system", "content": llm_shield_prompt},
            {"role": "user", "content": f"""<~START~>\n{llm_message}\n<~END~>"""},
        ]

        params = {
            "temperature": 0,
            "messages": messages,
            "request_timeout": 45,
        }

        if os.getenv("OPENAI_API_TYPE") == "azure":
            params["engine"] = os.getenv("OPENAI_API_DEPLOYMENT")
        else:
            params["model"] = shield_policy_config.get("model", settings.get("openai.openai_model", "gpt-3.5-turbo"))

        results = []
        response = openai.ChatCompletion.create(**params)
        response_val = response["choices"][0]["message"]["content"]
        parsed_dict = safe_loads_dictionary_string(response_val)
        if "true" == parsed_dict.get("eval", "true"):
            results = [CaptainsShieldDeflectionResult(llm_response=response_val, risk=1.0)]
        else:
            results = [CaptainsShieldDeflectionResult(llm_response=response_val, risk=0)]
        return results
