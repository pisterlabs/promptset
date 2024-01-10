from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from chains.action import fake_action_chain
from chains.query import fake_query_chain
from chains.general import general_chain


categorizer_chain = (
    {"user_input": RunnablePassthrough()}
    | ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template_file(
                template_file="./sys_input_categorizer.yaml", input_variables=[]
            ),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ]
    )
    | ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=1,
    )
    | StrOutputParser()
)


def category_router(x):
    if "action" in x["category"].lower():
        return fake_action_chain
    elif "query" in x["category"].lower():
        return fake_query_chain
    else:
        return general_chain.with_config(configurable={"memory": x["memory"]})


router_chain = (
    {
        "user_input": RunnablePassthrough(),
        "date": RunnablePassthrough(),
        "memory": RunnablePassthrough(),
        "long_term_memory": RunnablePassthrough(),
        "category": categorizer_chain,
    }
    | RunnableLambda(category_router)
    | StrOutputParser()
)
