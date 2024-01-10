import os

from langchain import LLMMathChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS


def get_response_from_query(query: str) -> str:
    if query is None:
        return ""

    print(f"query : {query}")
    # FAISS 먼저 적용하고 오기
    # docs = vector_db.similarity_search(query, k=k)
    embedding = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    path = "./DB/vector/korea_bank_700_information/index.faiss"
    print(os.getcwd())
    if os.path.exists(path):
        print(f"The file {path} exists.")
    else:
        print(f"The file {path} does not exist.")

    vector_db = FAISS.load_local("./DB/vector/korea_bank_700_information", embedding)
    docs = vector_db.similarity_search(query)
    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)

    template = """
    당신은 부동산을 구매하려는 사용자에게 금융, 부동산과 관련된 정보를 제공하는 assistant입니다.
    답변의 형식은 아래와 같이 진행합니다.

    "유저가 모르는 단어": "이에 대한 설명"
    "유저가 모르는 단어2": "이에 대한 설명2"
    
    Document retrieved from your DB : {docs}
    Answer the questions referring to the documents which you Retrieved from DB as much as possible.
    """
    # If you fell like you don't have enough-information to answer the question, say "제가 알고 있는 정보가 없습니다."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Answer the following question IN KOREAN: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(docs=docs, question=query)
    print(f"response = {response}")
    return response