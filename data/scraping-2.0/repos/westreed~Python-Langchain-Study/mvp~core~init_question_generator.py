from langchain import LLMChain
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)

from mvp.data_manager import *
from mvp.util import create_question_parser, remove_indent
from typing import *

QUESTION_COUNT = 10


def init_question_generator(
    data_manager: DataManager
) -> List:
    chat_manager = ChatManager()

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                remove_indent(
                    f"""You are an interviewer at {data_manager.company}.

                {data_manager.get_userdata()}
                """)),

            HumanMessagePromptTemplate.from_template(
                remove_indent(
                    """As an interviewer, you need to generate {question_count} interview questions based on the applicant's desired position, their cover letter. Additionally, consider the qualities and skills the company is looking for in candidates based on the job posting. Please follow the format below when creating the questions:
                
                    ```
                    1. Question content
                    2. Question content
                    3. Question content
                    ...
                    ```
                    
                    Please write in Korean.
                    """))
        ],
        input_variables=["question_count"],
    )

    create_question_chain = LLMChain(llm=chat_manager.get_chat_model(),
                                     prompt=prompt)
    output = create_question_chain(str(QUESTION_COUNT))
    return create_question_parser(output['text'])


"""
1. 지원자님은 백엔드 개발자로 지원한 이유가 무엇인가요? 어떤 측면에서 백엔드 개발에 매력을 느끼시나요?
2. 지금까지 개발해 본 프로젝트 중에서 특히 만족스러웠던 백엔드 개발 경험이 있으신가요? 그 이유는 무엇인가요?
3. 협업 경험이 부족하다고 언급하셨는데, 팀 프로젝트를 진행하면서 겪은 어려움과 그를 극복하는 방법에 대해 알려주세요.
4. SSAFY의 교육과정 중 백엔드 개발에 관련된 어떤 내용을 가장 기대하고 있으신가요?
5. 백엔드 개발에서의 기반 지식과 기술 스택 중에서 가장 자신 있는 부분은 무엇인가요? 그리고 그 부분을 어떻게 발전시키고 싶으신가요?
6. 삼성 청년 SW 아카데미에 지원하게 된 이유는 무엇인가요? SSAFY에서 얻고자 하는 가장 큰 이점은 무엇인가요?
7. 프로젝트를 진행할 때, 백엔드 개발자로서 중요하게 생각하는 가치나 원칙이 있으신가요? 어떤 가치를 추구하고 싶으신가요?
8. 백엔드 개발자로서 가장 도전적이었던 경험을 알려주세요. 그 도전을 극복한 데에 어떤 노력이나 전략이 있었나요?
9. SSAFY에서 함께 프로젝트를 진행하게 된다면, 팀원들과의 원활한 소통과 협업을 위해 어떤 노력을 기울이고 싶으신가요?
10. 백엔드 개발자로서 성장하기 위해 어떤 분야에서 더 발전하고 싶으신가요? 추가적인 기술 스택이나 역량을 키우기 위해 어떤 노력을 하고 계신가요?
"""
