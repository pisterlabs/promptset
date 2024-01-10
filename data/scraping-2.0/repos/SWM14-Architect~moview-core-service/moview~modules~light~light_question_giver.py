from typing import List, Optional
import openai
from moview.exception.light_question_parse_error import LightQuestionParseError
from moview.utils.prompt_loader import PromptLoader
from moview.environment.llm_factory import LLMModelFactory
from moview.config.loggers.mongo_logger import prompt_result_logger
from moview.utils.prompt_parser import PromptParser
from moview.utils.singleton_meta_class import SingletonMeta
from moview.decorator.retry_decorator import retry
from moview.utils.mixin.directory_mixin import DirectoryMixin


class LightQuestionGiver(DirectoryMixin, metaclass=SingletonMeta):

    def __init__(self, prompt_loader: PromptLoader):
        self.prompt = prompt_loader.load_prompt_json(LightQuestionGiver.__name__)
        openai.api_key = LLMModelFactory.load_api_key_for_open_ai()

    @retry()
    def give_light_questions_by_input_data(self, job_group: str, keyword: Optional[str], question_count: int) \
            -> List[str]:
        """

        Args:
            job_group: 직군
            keyword: 직무 면접 키워드 문자열
            question_count: 출제할 질문 개수

        Returns: 직무 중심으로 출제된 질문 리스트 (light mode 전용)

        """

        model = "gpt-3.5-turbo-16k"

        human_message = "양식을 지켜서 직무 기술 면접 질문을 생성하세요." if keyword is None \
            else f" 양식을 지켜서 {keyword}에 대한 직무 기술 면접 질문을 생성하세요."

        messages = [{
            "role": "system",
            "content": self.prompt.format(job_group=job_group, question_count=question_count)
        }, {
            "role": "user",
            "content": human_message
        }]

        response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.7)

        prompt_result = response['choices'][0]['message']['content']

        prompt_result_logger("light question prompt result", prompt_result=prompt_result)

        parse_question = PromptParser.parse_question(prompt_result)

        if parse_question is not None:
            return parse_question
        else:
            raise LightQuestionParseError()
