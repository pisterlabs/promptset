from typing import List
import openai
from moview.exception.initial_question_parse_error import InitialQuestionParseError
from moview.utils.prompt_loader import PromptLoader
from moview.environment.llm_factory import LLMModelFactory
from moview.config.loggers.mongo_logger import prompt_result_logger
from moview.utils.prompt_parser import PromptParser
from moview.decorator.retry_decorator import async_retry
from moview.utils.singleton_meta_class import SingletonMeta


class InitialQuestionGiver(metaclass=SingletonMeta):

    def __init__(self, prompt_loader: PromptLoader):
        self.prompt = prompt_loader.load_prompt_json(InitialQuestionGiver.__name__)
        openai.api_key = LLMModelFactory.load_api_key_for_open_ai()

    @async_retry()
    async def give_initial_questions_by_input_data(
            self, company_name, recruit_announcement: str, cover_letter: str, question_count: int, exclusion_list: List[str] = None
    ) -> List[str]:
        """
        Args:
            company_name: 회사 이름
            recruit_announcement: 모집공고
            cover_letter: 자기소개서
            question_count: 출제할 질문 개수
            exclusion_list: 제외할 질문 리스트
        Returns: 생성된 질문 리스트 (자기소개서와 모집공고가 포함되어 있기 때문에 개인맞춤형 질문)
        """
        exclusion_question = self.__create_exclusion_question_string(exclusion_list)

        model = "gpt-3.5-turbo-16k"

        messages = [{
            "role": "system",
            "content": self.prompt["create_question_by_input_data"].format(
                company_name=company_name,
                exclusion_question=exclusion_question,
                question_count=question_count)
        }, {
            "role": "user",
            "content": f"양식을 지켜서 질문을 생성하세요.\n\n[회사의 모집공고]\n{recruit_announcement}\n\n[면접자의 자기소개서]\n{cover_letter}"
        }]

        response = await openai.ChatCompletion.acreate(model=model, messages=messages, temperature=0.7)

        initial_questions_from_llm = response['choices'][0]['message']['content']

        prompt_result_logger("initial question (input data) prompt result", prompt_result=initial_questions_from_llm)

        parse_question = self.__parse_result_from_llm(initial_questions_from_llm)
        # 파싱된 질문 개수가 출제할 질문 개수와 같으면, 파싱 성공으로 간주합니다. 파싱이 성공하면, 파싱된 질문 리스트를 반환합니다.
        if len(parse_question) == question_count:
            return parse_question
        else:
            raise InitialQuestionParseError()  # 파싱이 실패하면, InitialQuestionParseError를 발생시킵니다.

    @async_retry()
    async def give_initial_questions(
            self, company_name:str, job_group: str, question_count: int, exclusion_list: List[str] = None
    ) -> List[str]:
        """
        Args:
            company_name: 회사 이름
            job_group: 타겟 직군
            question_count: 출제할 질문 개수
            exclusion_list: 제외할 질문 리스트
        Returns: 생성된 질문 리스트 (자기소개서와 모집공고가 없기 때문에 해당 직군에 대한 광범위 질문)
        """
        exclusion_question = self.__create_exclusion_question_string(exclusion_list)

        model = "gpt-3.5-turbo-16k"

        messages = [{
            "role": "system",
            "content": self.prompt["create_question"].format(
                company_name=company_name,
                exclusion_question=exclusion_question,
                job_group=job_group,
                question_count=question_count)
        }, {
            "role": "user",
            "content": "양식을 지켜서 질문을 생성하세요"
        }]

        response = await openai.ChatCompletion.acreate(model=model, messages=messages, temperature=0.7)

        initial_questions_from_llm = response['choices'][0]['message']['content']

        prompt_result_logger("initial question prompt result", prompt_result=initial_questions_from_llm)

        parse_question = self.__parse_result_from_llm(initial_questions_from_llm)
        # 파싱된 질문 개수가 출제할 질문 개수와 같으면, 파싱 성공으로 간주합니다. 파싱이 성공하면, 파싱된 질문 리스트를 반환합니다.
        if parse_question is not None:
            return parse_question
        else:
            raise InitialQuestionParseError()  # 파싱이 실패하면, InitialQuestionParseError를 발생시킵니다.

    @staticmethod
    def __create_exclusion_question_string(exclusion_list: List[str]) -> str:
        """
        Args:
            exclusion_list: 제외할 질문 리스트
        Returns: 제외할 질문 리스트를 하나의 문자열로 합친 결과
        """
        exclusion_question = ""
        if exclusion_list is not None:
            for idx, question in enumerate(exclusion_list):
                exclusion_question += f"{idx + 1}. {question}\n"
        return exclusion_question

    @staticmethod
    def __parse_result_from_llm(initial_questions_from_llm: str) -> List[str]:
        """
        Args:
            initial_questions_from_llm: llm으로부터 온 초기 질문 문자열
        Returns:  초기 질문 문자열 리스트 (파싱됨)
        """
        return PromptParser.parse_question(initial_questions_from_llm)
