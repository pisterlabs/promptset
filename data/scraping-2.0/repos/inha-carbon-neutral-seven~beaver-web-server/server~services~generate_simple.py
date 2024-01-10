import logging

from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ..models.generate import Answer, Question
from .ping import check_server_status


async def generate_message(question: Question) -> Answer:
    """
    unused:

    """
    if await check_server_status() is False:  # 모델 서버가 불안정하면 임베딩을 진행하지 않음
        return Answer(message="모델 서버가 불안정합니다. 나중에 시도해주세요. ")

    # "첨부한 자료를 근거로 해서 질문에 답해주시기 바랍니다." 문장 일단 제외
    template = """### Prompt:
당신은 AI 챗봇이며, 사용자에게 도움이 되는 유익한 내용을 제공해야 합니다.
index를 적극적으로 활용하여 질문에 답해주시기 바랍니다.
### Question:
{question}
### Answer:"""

    prompt = PromptTemplate(
        input_variables=["question"],
        template=template,
    )
    llm_chain = LLMChain(
        llm=OpenAI(temperature=0.1),
        prompt=prompt,
        verbose=False,
    )
    res = llm_chain.predict(question=question.message)
    answer = Answer(message=res)

    logging.info("생성한 응답: %s", answer.message)
    return answer
