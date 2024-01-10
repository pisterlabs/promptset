from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

def get_answer(chunks, question):       
    docs = chunks.similarity_search(question)
    human_message_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="You",
                input_variables=["product"],
            )
        )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    llm = ChatOpenAI(temperature=0)

    chain = LLMChain(llm=llm, prompt=chat_prompt_template)
    print(chain.run("colorful socks"))





