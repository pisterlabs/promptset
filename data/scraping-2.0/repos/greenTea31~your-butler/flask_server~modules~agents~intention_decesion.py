from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


def decide(chat: str):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)

    _DEFAULT_TEMPLATE = """
    당신은 사용자의 입력을 읽고 그에 대한 의도를 파악하는 assistant입니다.
    사용자의 채팅을 보고 다음과 같은 3가지 채팅으로 분류하여 출력해야 합니다.
    
    1. 사용자가 대출 상품의 정보를 얻기를 원하는 의도의 질문
    
    2. 사용자가 대출 상품 외의, 금융 용어에 대한 정보를 얻기를 원하는 의도의 질문
    
    3. 그 외의 질문
    
    대답의 형태는 반드시 아래와 같아야 합니다. 큰 따옴표에 유의하십시오. 반드시 JSON으로 Parsing이 가능한 형태여야 합니다.
    
    {{
        "loan": 대출 상품의 정보를 얻기를 원하는 의도의 질문이 들어갑니다,
        "finance": 금융 용어에 대한 정보를 얻기를 원하는 의도의 질문이 들어갑니다,
        "other": 그 외의 질문이 들어갑니다.
    }}
    
    사용자의 채팅 정보는 아래와 같습니다.
    
    {chat}
    """

    prompt_template = PromptTemplate(
        input_variables=["chat"], template=_DEFAULT_TEMPLATE
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(chat)
    print(f"decide response = {response}")
    return response

