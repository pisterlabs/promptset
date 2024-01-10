import json
from typing import List, Dict, Any, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain

from src.configuration import logger
from src.llm_module.llms import OpenAI3Model, OpenAI4Model
from src.llm_module.prompts import course_prompt


class LlmTypes:
    Gpt_3 = 'gpt_3'
    Gpt_4 = 'gpt_4'


class CourseChain(Chain):
    llms = {
        LlmTypes.Gpt_3: OpenAI3Model(),
        LlmTypes.Gpt_4: OpenAI4Model()
    }

    @property
    def input_keys(self) -> List[str]:
        return ["query", "language"]

    @property
    def output_keys(self) -> List[str]:
        return ["course_content"]

    def _get_llm_version(self, llm_version):
        llm = self.llms.get(llm_version)
        if llm is None:
            raise ValueError(f"Invalid llm_version: {llm_version}")
        return llm

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict:
        llm_version = self._get_llm_version(inputs.get("llm_version"))
        query = inputs["query"]
        language = inputs["language"]

        prompt_value = course_prompt.format_prompt(query=query, language=language)

        logger.info(f"Full prompt: `{prompt_value}`")

        response = llm_version.generate_prompt(
            [prompt_value],
            callbacks=run_manager.get_child() if run_manager else None,
        )

        response_text = response.generations[0][0].text

        result = {
            "course_content": json.loads(response_text.replace("```json", "").replace("```", ""))
        }

        logger.info(f"Course generation for '{query}' is done.")

        return result
