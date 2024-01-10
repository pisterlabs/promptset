from typing import Any, Dict, List, cast

from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline, Replicate
from langchain.prompts.chat import ChatPromptTemplate, ChatPromptValue, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, PromptValue, SystemMessage

from credit_hacking.llms.registered_llm import RegisteredLLM

FIXED_SYSTEM_PROMPT = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>"""  # noqa: E501


class Llama2ChatPromptValue(ChatPromptValue):
    def to_string(self) -> str:
        """Convert sequence of Messages to Llama 2 chat prompt format."""
        formatted_messages = []
        user_system_prompt: str = ""
        for m in self.messages:
            if isinstance(m, SystemMessage):
                user_system_prompt = m.content
            elif isinstance(m, HumanMessage) or isinstance(m, AIMessage):
                formatted_messages.append(f"{m.content} [/INST]")
            else:
                raise ValueError(f"Got unsupported message type: {m}")

        return f"{FIXED_SYSTEM_PROMPT}\n\n{user_system_prompt}\n{' '.join(formatted_messages)}"


class Llama2ChatPromptTemplate(ChatPromptTemplate):
    input_variables: List[str] = ["stop"]

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        messages = self.format_messages(**kwargs)
        return Llama2ChatPromptValue(messages=messages)


class Llama2ChatChain(LLMChain):
    def prep_inputs(self, inputs: Dict[str, Any] | Any) -> Dict[str, str]:
        inputs = cast(Dict[str, Any], super().prep_inputs(inputs))
        inputs["stop"] = inputs.get("stop", [])
        inputs["stop"].append(r"\[\/INST\]")
        return inputs


class Llama27BChatLLM(RegisteredLLM):
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    temperature = 0
    load_in_8bit = True
    max_length = 4096

    @classmethod
    def get_llm(cls, system_prompt: str) -> LLMChain:
        system_message = SystemMessage(content=system_prompt)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")

        chat_prompt = Llama2ChatPromptTemplate.from_messages([system_message, human_message_prompt])

        llm = HuggingFacePipeline.from_model_id(
            model_id=cls.model_id,
            task="text-generation",
            model_kwargs={
                "temperature": cls.temperature,
                "load_in_8bit": cls.load_in_8bit,
                "max_length": cls.max_length,
            },
        )
        chain = Llama2ChatChain(llm=llm, prompt=chat_prompt)

        return chain

    @classmethod
    def get_cli_name(cls) -> str:
        return "llama-2-7b-chat"


class Llama270BChatLLM(RegisteredLLM):
    model_id = "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781"
    temperature = 0.01  # Minimum required by Replicate
    max_length = 4096

    @classmethod
    def get_llm(cls, system_prompt: str) -> LLMChain:
        system_message = SystemMessage(content=system_prompt)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")

        chat_prompt = Llama2ChatPromptTemplate.from_messages([system_message, human_message_prompt])

        llm = Replicate(
            model=cls.model_id,
            input={
                "temperature": cls.temperature,
                "max_length": cls.max_length,
            },
        )
        chain = Llama2ChatChain(llm=llm, prompt=chat_prompt)

        return chain

    @classmethod
    def get_cli_name(cls) -> str:
        return "llama-2-70b-chat"
