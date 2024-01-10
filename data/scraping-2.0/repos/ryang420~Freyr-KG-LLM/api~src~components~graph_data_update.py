from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.openai_functions import (
    create_structured_output_chain,
)

from langchain_core.output_parsers import StrOutputParser


def graph_data_augmentation(llm: ChatOpenAI,
                            graph_data: str,
                            user_input: str):
    """Augment graph data with user input."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            Your task is to make changes on the graph data by user's input and return a json string.
            The graph data is a json format string. It contains two keys: nodes and links. 
            The user's input is to add, update or delete the nodes and links. 
            If the deleted node has any relationships or links on it, delete them as well.
            Return me the json format string of the updated graph data only. No additional information included!

            """
        ),
        ("human", "Use the given context to update {graph_data} by {user_input}")
    ])

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    return chain.invoke({"graph_data": graph_data, "user_input": user_input})
