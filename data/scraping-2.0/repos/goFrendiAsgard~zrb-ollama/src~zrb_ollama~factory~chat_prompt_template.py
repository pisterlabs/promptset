from langchain.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder,
    SystemMessagePromptTemplate
)
from .schema import ChatPromptTemplateFactory
from ..task.any_prompt_task import AnyPromptTask


def chat_prompt_template_factory() -> ChatPromptTemplateFactory:
    def create_chat_prompt_template(task: AnyPromptTask) -> ChatPromptTemplate:
        messages = []
        # Add system prompt to message
        rendered_system_prompt = task.get_rendered_system_prompt()
        if rendered_system_prompt is not None:
            messages.append(SystemMessagePromptTemplate.from_template(
                rendered_system_prompt
            ))
        # Add chat history and human message
        messages = messages + [
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        # Create chat prompt template
        return ChatPromptTemplate(messages=messages)
    return create_chat_prompt_template
