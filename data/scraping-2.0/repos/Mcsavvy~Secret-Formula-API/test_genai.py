from typing import Any, Dict

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from cookgpt.ext.config import config  # noqa: F401


class Memory(ConversationBufferMemory):
    input_key: str = "input"

    @property
    def memory_variables(self) -> list[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key, "name"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            self.memory_key: self.buffer_as_messages,
        }


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are an AI named {name}"
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("Hi {input}"),
    ]
)
memory = Memory(input_key="input", output_key="response")
llm = ChatGoogleGenerativeAI(  # type: ignore[call-arg]
    model="gemini-pro", convert_system_message_to_human=True
)
chain = ConversationChain(llm=llm, prompt=prompt, memory=memory)
result = chain.invoke(
    dict(name="Bard", input="I'm doing well, how are you?"),
)
print(result)
chain.__call__
# contents = content_types.to_contents(content)
# print(f"content: {contents}")
# model: GenerativeModel = llm._generative_model
# model.generate_content
# print(f"prompt: {model.co}")
# client: GenerativeServiceClient = model._client
# count = client.count_tokens(contents=contents, model=model.model_name)
# print(f"count: {count.total_tokens}")
