from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField, RunnablePassthrough


general_chain = (
    {
        "user_input": RunnablePassthrough(),
        "long_term_memory": RunnablePassthrough(),
        "date": RunnablePassthrough(),
    }
    | ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template_file(
                template_file="./prompts/sys_general_chain.yaml",
                input_variables=["date", "long_term_memory"],
            ),
            MessagesPlaceholder(variable_name="recent_messages"),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ]
    )
    | ChatOpenAI(model="gpt-4").configurable_fields(
        memory=ConfigurableField(
            id="memory",
        )
    )
    | StrOutputParser()
)
