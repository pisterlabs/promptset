from typing import List
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel


class MemoryCategory(BaseModel):
    key: str


class MemoryCategories(BaseModel):
    categories: List[MemoryCategory]


input_to_memory_category_list = (
    {
        "user_input": RunnablePassthrough(),
        "date": RunnablePassthrough(),
        "memory_schema": RunnablePassthrough(),
    }
    | ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template_file(
                template_file="./prompts/sys_input_to_memory_categories.yaml",
                input_variables=["date", "memory_schema"],
            ),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ]
    )
    | ChatOpenAI(model="gpt-4", temperature=0.05, max_tokens=256)
    | PydanticOutputParser(pydantic_object=MemoryCategories)
)

input_to_memory_category = (
    {
        "user_input": RunnablePassthrough(),
        "date": RunnablePassthrough(),
        "memory_schema": RunnablePassthrough(),
    }
    | ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template_file(
                template_file="./prompts/sys_input_to_memory_category.yaml",
                input_variables=["date", "memory_schema"],
            ),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ]
    )
    | ChatOpenAI(model="gpt-4", temperature=0.05, max_tokens=15)
    | PydanticOutputParser(pydantic_object=MemoryCategory)
)


memory_synthesizer = (
    {"old_memory": RunnablePassthrough(), "new_memory": RunnablePassthrough()}
    | ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template_file(
                template_file="./prompts/user_memory_synthesizer.yaml",
                input_variables=["old_memory", "new_memory"],
            )
        ]
    )
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)
    | StrOutputParser()
)
