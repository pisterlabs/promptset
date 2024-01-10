import json
import pathlib
from typing import List

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from gembox.format_utils import to_sorted_list_str
from gembox.collection_utils import is_empty_collection
from tenacity import retry, stop_after_attempt, retry_if_exception_type

from ..schema import LLMResponse, Example

config_path = pathlib.Path(__file__).parent / ".autom" / "developer.json"
with open(config_path, "r") as f:
    developer_config_data: dict = json.load(f)


class Developer:
    """
    This is the abstract class for a GPT-based developer.

    A well-defined developer should be defined by:
    - role: The role the developer plays.
    - work_output_requirement: The type of work result the developer needs to produce.

    The developer always should convey the following as their working result:
    - code: The code that can accomplish the user's task.
    - explanation: The explanation of the code, which can be used to explain the code to the user.
    """

    def __init__(self):
        self._llm = ChatOpenAI(model=self.model)

    @retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception))
    def serve(self, user_msg: str):
        """
        Serve the user's message and set up the work result.
        """
        # 1. Query the llm
        llm_resp_text = self._query_llm(user_msg)

        # 2. Parse the llm response
        llm_resp = self._parse_llm_resp(llm_resp_text)

        # 3. return the result
        return llm_resp

    def _query_llm(self, user_msg: str) -> str:
        return self.llm.predict(self.template.format(user_message=user_msg))

    def _parse_llm_resp(self, llm_resp_text: str) -> LLMResponse:
        code = self._get_code_from_llm_resp(llm_resp_text)
        explanation = self._get_explanation_from_llm_resp(llm_resp_text)
        return LLMResponse(code=code, explanation=explanation)

    def _get_code_from_llm_resp(self, llm_resp_text: str) -> str:
        """parse the code from the llm response"""
        code_start, code_end = "MAGIC$$$", "$$$"
        code = llm_resp_text.split(code_start)[1].split(code_end)[0].strip()
        return code

    def _get_explanation_from_llm_resp(self, llm_resp_text: str) -> str:
        """parse the explanation from the llm response"""
        explanation_start, explanation_end = "MAGIC%%%", "%%%"
        explanation = llm_resp_text.split(explanation_start)[1].split(explanation_end)[0].strip()
        return explanation

    @property
    def template(self) -> PromptTemplate:
        template_str = ""
        if self.job_description is not None:
            template_str += (self.job_description + "\n")

        requirements = self.requirements
        if not is_empty_collection(requirements):
            template_str += "Requirements:\n"
            template_str += to_sorted_list_str(list(requirements))

        # add examples
        examples = self.examples
        if not is_empty_collection(examples):
            template_str += "\n"
            for i, example in enumerate(examples):
                template_str += f"e.g.{i + 1}\n[Query]:\n{example.query}\n[Response]:\n{example.response}\n"
            template_str += "Now, it's your turn.\n"

        template_str += "[Query]:\n{user_message}\n[Response]:"

        return PromptTemplate.from_template(template_str)

    @property
    def model(self) -> str:
        return developer_config_data["model"]

    @property
    def role(self) -> str:
        return developer_config_data["title"]

    @property
    def job_description(self) -> str:
        # if no key, then return None
        return developer_config_data["job-description"]

    @property
    def guidance_requirements(self) -> List[str]:
        requirements = developer_config_data.get("extra-guidance-requirements", None)
        if requirements is None:
            return []
        return requirements

    @property
    def format_requirements(self) -> List[str]:
        requirements = developer_config_data.get("extra-format-requirements", None)
        if requirements is None:
            return []
        return requirements

    @property
    def examples(self) -> List[Example]:
        example_dict_list = developer_config_data.get("examples", None)
        if example_dict_list is None:
            return []
        return [Example(example_data["query"], example_data["response"]) for example_data in
                developer_config_data["examples"]]

    @property
    def system_message(self) -> str:
        return f"You are a {self.role} to translate user's needs into high quality code."

    @property
    def requirements(self) -> List[str]:
        """returns the requirement of the work output, including the requirements for guidance and formatting"""
        return self.guidance_requirements + self.format_requirements

    @property
    def llm(self):
        """get the llm instance"""
        return self._llm


__all__ = ['Developer']
