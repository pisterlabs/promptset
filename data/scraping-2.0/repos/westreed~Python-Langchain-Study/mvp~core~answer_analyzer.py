from langchain import LLMChain, ConversationChain
from langchain.chains import MultiPromptChain
from langchain.chains.router.llm_router import (
    RouterOutputParser,
    LLMRouterChain,
)
from langchain.chains.router.multi_prompt_prompt import (
    MULTI_PROMPT_ROUTER_TEMPLATE,
)
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.prompts import PromptTemplate

from mvp.data_manager import *
from mvp.util import remove_indent
from typing import *


def make_specific_prompt_with_knowledge(
    data_manager: DataManager,
    question: str,
) -> List[Dict]:
    fit_feature_dict = {
        "job_fit": ("직무 적합성", "직무를 수행하는 데 필요한 기술,지식, 그리고 경험을 가지고 있는지 평가하는 것"),
        "cultural_fit": ("문화 적합성", "조직의 가치와 문화에 잘 맞는지 평가하는 것"),
        "project_management": (
            "프로젝트 관리 능력",
            "특정 프로젝트를 기획하고, 이를 성공적으로 실행하고, 필요한 변경 사항을 관리하는지 평가하는 것"),
        "communication": (
            "의사소통 능력", "자신의 아이디어를 명확하게 전달하고, 다른 사람들과 효과적으로 협업할 수 있는지 평가하는 것"),
        "personality": ("인성 및 태도", "성격, 성실성, 성장 마인드셋을 평가하는 것"),
        "motivation": ("열정 및 지원동기", "왜 그 직무를 선택하고, 그 회사에서 일하길 원하는지 평가하는 것"),
        "adaptability": ("적응력", "새로운 환경이나 상황에 얼마나 빠르게 적응하는지를 평가하는 것"),
        "learning_ability": (
            "학습 능력", "지식이나 기술을 빠르게 습득하고 새로운 정보를 효과적으로 사용하는 지 평가하는 것"),
        "leadership": ("리더십", "팀에서 리더로서 역할을 수행한 경험이나 리더십에 대한 지식을 평가하는 것")
    }

    knowledge_prompt = remove_indent(
        """{review_standard_detail}을 {review_standard_knowledge}이라 합니다.
        면접관인 당신은 {review_standard_knowledge}의 관점에서 면접자의 답변을 평가해야 합니다."""
    )

    # 분석 체인은 라우팅 체인이므로 라우터 적용.
    prompt_infos = []

    for fit_feature in fit_feature_dict:
        prompt_info = {
            "name": fit_feature,
            "description": knowledge_prompt.format(
                review_standard_knowledge=fit_feature_dict[fit_feature][0],
                review_standard_detail=fit_feature_dict[fit_feature][1],
            ),
            "prompt_template": remove_indent(
                f"""You are an interviewer.
                As an interviewer, please analyze the interviewee's response and provide evaluations by dividing them into positive aspects and areas for improvement. When mentioning areas for improvement, please focus only on the truly disappointing aspects. Please follow the format below:

                ```
                '좋은점':
                - Positive aspect content
                
                '아쉬운점':
                - Areas for improvement content
                ```
                Furthermore, the following content includes company information and the applicant's self-introduction.
                {data_manager.get_userdata()}
                
                The question and the candidate's response are as follows:
                ```
                Interviewer`s Question:
                {question}
                Interviewee`s Answer:""" +
                remove_indent("""{input}
                ```
                
                Please write in Korean.""")
            ),
        }
        prompt_infos.append(prompt_info)
    return prompt_infos


def make_router_chain(llm: ChatOpenAI, prompt_infos: List) -> LLMRouterChain:
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
        destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    return LLMRouterChain.from_llm(llm, router_prompt)


def make_destination_chains(llm: ChatOpenAI, prompt_infos: List) -> Dict[str, LLMChain]:
    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["input"])
        destination_chains[name] = LLMChain(llm=llm, prompt=prompt)
    return destination_chains


def answer_analyzer(
    data_manager: DataManager,
    question_entity: QuestionEntity,
    evaluation_manager: EvaluationManager,
):
    question = question_entity.question
    answer = question_entity.answer
    llm = ChatManager().get_chat_model()

    prompt_infos = make_specific_prompt_with_knowledge(data_manager, question)
    router_chain = make_router_chain(llm=llm, prompt_infos=prompt_infos)
    default_chain = ConversationChain(llm=llm, output_key="text")
    destination_chains = make_destination_chains(llm=llm, prompt_infos=prompt_infos)

    chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
    )

    evaluation = chain.run(answer)

    result = remove_indent(
        f"""면접관 질문:
        {question}
        
        면접자 답변:
        {answer}
        
        면접관 평가:
        {evaluation}
        """
    )
    evaluation_manager.add_answer_evaluation(result)
    return result

