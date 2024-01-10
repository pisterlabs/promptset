from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField, RunnablePassthrough


action_succeeded_format_chain = (
    {
        "user_input": RunnablePassthrough(),
        "long_term_memories": RunnablePassthrough()
    }
    | ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template_file(
                template_file="./prompts/user_aifred_action_succesful_format.yaml",
                input_variables=["user_input", "long_term_memories"],
            ),
        ]
    )
    | ChatOpenAI(model="gpt-3.5-turbo").configurable_fields(
        memory=ConfigurableField(
            id="memory",
        )
    )
    | StrOutputParser()
)

action_failed_format_chain = (
    {
        "user_input": RunnablePassthrough(),
        "long_term_memories": RunnablePassthrough(),
        "action_output": RunnablePassthrough()
    }
    | ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template_file(
                template_file="./prompts/user_aifred_action_succesful_format.yaml",
                input_variables=["user_input", "long_term_memories", "action_output"],
            ),
        ]
    )
    | ChatOpenAI(model="gpt-3.5-turbo").configurable_fields(
        memory=ConfigurableField(
            id="memory",
        )
    )
    | StrOutputParser()
)

query_format_chain = (
    {
        "user_input": RunnablePassthrough(),
        "long_term_memories": RunnablePassthrough(),
        "query_result": RunnablePassthrough(),
    }
    | ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template_file(
                template_file="./prompts/sys_aifred_query_format.yaml",
                input_variables=["user_input", "long_term_memories", "query_result"],
            ),
            MessagesPlaceholder(variable_name="recent_messages"),
            HumanMessagePromptTemplate.from_template("{user_input}"),
            AIMessagePromptTemplate.from_template("Query result:\n{query_result}")
        ]
    )
    | ChatOpenAI(model="gpt-4").configurable_fields(
        memory=ConfigurableField(
            id="memory",
        )
    )
    | StrOutputParser()
)
