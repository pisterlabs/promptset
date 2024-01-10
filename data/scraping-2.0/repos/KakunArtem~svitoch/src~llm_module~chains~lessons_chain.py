from typing import List, Dict, Any, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel

from src.llm_module.chains.course_chain import LlmTypes
from src.configuration import logger
from src.llm_module.llms import OpenAI4Model, OpenAI3Model
from src.llm_module.prompts import lessons_template


class LessonsChain(Chain):
    llms = {
        LlmTypes.Gpt_3: OpenAI3Model(),
        LlmTypes.Gpt_4: OpenAI4Model()
    }

    @property
    def input_keys(self) -> List[str]:
        return ["course_content", "language"]

    @property
    def output_keys(self) -> List[str]:
        return ["lessons_content"]

    def _get_llm_version(self, llm_version):
        llm = self.llms.get(llm_version)
        if llm is None:
            raise ValueError(f"Invalid llm_version: {llm_version}")
        return llm

    def _get_lessons(self, lessons):
        return [lesson for lesson in lessons.get("lessons")]

    def _get_lessons_dict(self, lessons):
        return {f"key{i + 1}": lesson for i, lesson in enumerate(lessons)}

    def _get_prompts_list(self, topics, llm):
        llm_version = self._get_llm_version(llm)
        return {
            f"key{i + 1}": ChatPromptTemplate.from_template(
                lessons_template.replace("lessons", f"key{i + 1}")
            ) | llm_version for i in range(len(topics))
        }

    def _call(self,
              inputs: Dict[str, Any],
              run_manager: Optional[CallbackManagerForChainRun] = None,
              ):
        lessons = self._get_lessons(inputs.get("course_content"))
        topics_dict = self._get_lessons_dict(lessons)
        prompt_list = self._get_prompts_list(lessons, inputs.get("llm_version"))

        response = RunnableParallel(prompt_list).invoke({
            **topics_dict,
            "language": inputs.get("language"),
            "context": inputs.get("course_content").get("course_name")
            })

        logger.info(f"Lessons generation is done.")

        return {
            "course_name": inputs.get("course_content").get("course_name"),
            "lessons_content": {key: value.content for key, value in response.items()}
        }
