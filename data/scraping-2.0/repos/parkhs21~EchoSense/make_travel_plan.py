from dotenv import load_dotenv
from pathlib import Path
import os

#api키를 .env파일에서 로드
def load_secret():
    load_dotenv()
    env_path = Path('.') / '.env'
    load_dotenv(dotenv_path=env_path)
    
    open_ai_key = os.getenv("OPENAI_API_KEY")
    google_palm_key = os.getenv("GOOGLE_PALM_API_KEY")
    
    return {
        "OPENAI_API_KEY": open_ai_key,
    }

#쿼리의 유효성 검사
#여행 일정 제작
#Google Maps API에서 이해할 수 있는 형식으로 경유지를 추출

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

#유효성 검사
class Validation(BaseModel):
    plan_is_valid: str = Field(description="일정이 유효하면 'yes', 유효하지 않으면 'no'를 반환합니다.")
    updated_request: str = Field(description="새로운 일정을 반환합니다.")

class ValidationTemplate(object):
    def __init__(self):
        self.system_template = """
        사용자가 흥미로운 여행 계획을 세울 수 있도록 도와주는 agent입니다.

      사용자의 요청은 4개의 해시태그로 표시됩니다. 사용자의 요청이 설정한 제약 조건 내에서
      요청이 합리적이고 사용자가 설정한 제약 조건 내에서 달성 가능한지 판단합니다.

      유효한 요청에는 다음이 포함되어야 합니다:
      - 시작 및 종료 위치
      - 시작 및 종료 위치를 고려할 때 합리적인 여행 기간
      - 사용자의 관심사 및/또는 선호하는 교통수단과 같은 기타 세부 정보

      요청이 유효하지 않은 경우
      plan_is_valid = 0으로 설정하고 여행 전문 지식을 활용하여 요청을 유효하도록 업데이트하세요,
      수정된 요청은 100단어 미만으로 유지하세요.

      요청이 타당하다고 판단되면 plan_is_valid = 1로 설정하고
      요청을 수정하지 마세요.
      
      {format_instruction}
      """
        self.human_template = """####{query}####"""
        
        self.parser = PydanticOutputParser(pydantic_object=Validation)
        
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template,
            partial_variables={"format_instruction": self.parser.get_format_instructions()},
        )
        
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template,
            input_variables=["query"]
        )
        
        self.chat_prompt = ChatPromptTemplate.from_messages([self.system_message_prompt,self.human_message_prompt])
        
import openai
import logging
import time

from langchain.llms import GooglePalm
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain

#logging.basicConfig(level=logging.INFO) 
#llm api를 호출하는 agent class(chatOpenAI사용)
class Agent(object):
    def __init__(
        self,
        open_ai_api_key,
        model="gpt-4-1106-preview",
        temperature=0,
        debug=True,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self._openai_key = open_ai_api_key
        self.chat_model = ChatOpenAI(model=model,
                                     temperature=temperature,
                                     openai_api_key=self._openai_key)
        self.validation_prompt = ValidationTemplate()
        self.itinerary_prompt = ItineraryTemplate()
        
        self.validation_chain = self._set_up_validation_chain(debug)
        self.agent_chain = self._set_up_agent_chain(debug)
        
    def _set_up_validation_chain(self, debug=True):
        # make validation agent chain
        validation_agent = LLMChain(
            llm=self.chat_model,
            prompt=self.validation_prompt.chat_prompt,
            output_parser=self.validation_prompt.parser,
            output_key="validation_output",
            verbose=debug,
        )
        
        # add to sequential chain 
        overall_chain = SequentialChain(
            chains=[validation_agent],
            input_variables=["query", "format_instructions"],
            output_variables=["validation_output"],
            verbose=debug,
        )

        return overall_chain
    
    def _set_up_agent_chain(self, debug=True):
        travel_agent = LLMChain(
            llm=self.chat_model,
            prompt=self.itinerary_prompt.chat_prompt,
            verbose=debug,
            output_key="agent_suggestion",
        )
        
        return travel_agent
    
    def suggest_travel(self, query):
        #self.logger.info("Validating query")
        t1 = time.time()
        self.logger.info(
            "Calling validation (model is {}) on user input".format(
                self.chat_model.model_name
            )
        )
        validation_result = self.validation_chain(
            {
                "query": query,
                "format_instructions": self.validation_prompt.parser.get_format_instructions(),
            }
        )

        validation_test = validation_result["validation_output"].dict()
        t2 = time.time()
        #self.logger.info("Time to validate request: {}".format(round(t2 - t1, 2)))
        
        if validation_test["plan_is_valid"].lower() == "no":
            self.logger.warning("User request was not valid!")
            print("\n######\n Travel plan is not valid \n######\n")
            print(validation_test["updated_request"])
            return None, None, validation_result

        else:
            # plan is valid
            self.logger.info("Query is valid")
            self.logger.info("Getting travel suggestions")
            t1 = time.time()

            self.logger.info(
                "User request is valid, calling agent (model is {})".format(
                    self.chat_model.model_name
                )
            )

            agent_result = self.agent_chain({"query": query})

            trip_suggestion = agent_result["agent_suggestion"]
            
            return trip_suggestion, validation_test


class ItineraryTemplate(object):
    def __init__(self):
        self.system_template = """
      사용자가 흥미로운 여행 계획을 세울 수 있도록 도와주는 agent입니다.

      사용자의 요청은 4개의 해시태그로 표시됩니다. 사용자의 요청을
      사용자의 요청을 방문해야 할 장소와 해야 할 일을 설명하는 세부 여정으로
      방문해야 할 장소와 해야 할 일을 설명하는 세부 일정으로 변환합니다.

      각 장소의 구체적인 주소를 포함하도록 합니다.

      사용자의 선호도와 시간대를 고려하고, 제약 조건을 고려할 때 재미있고 실행 가능한 일정을 제공한다.

      여정은 명확한 시작 및 종료 위치가 포함된 글머리 기호 목록으로 제공한다.
      출력물은 목록만 작성하고 다른 것은 작성하지 않아야 한다.
    """

        self.human_template = """
      ####{query}####
    """

        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template,
        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["query"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )
        
secrets = load_secret()
travel_agent = Agent(open_ai_api_key=secrets['OPENAI_API_KEY'],debug=True)

query = """
        제주도에서 3박 4일 여행을 계획하고 있습니다. 한라산을 등반하고 싶습니다. 제주도 해변도 가고 싶습니다.
        """

plan, validation_result = travel_agent.suggest_travel(query)
print(plan)
print(validation_result)
