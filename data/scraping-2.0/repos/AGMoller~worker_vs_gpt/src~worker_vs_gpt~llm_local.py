from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain import LLMChain

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-2-70b-chat-hf",
    task="text-generation",
    model_kwargs={
        "temperature": 0.7,
        "max_length": 1024,
        "load_in_8bit": True,
    },
)


def get_empathy_prompt() -> ChatPromptTemplate:
    system_message = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=[],
            template="""
        You are an advanced AI writer. Your job is to help write examples of texts that convey politeness or not.
        """,
        )
    )

    human_message = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["text", "politeness"],
            template="""The following text has a {politeness} flag for expressing politeness, write 9 new semantically similar examples that show the same intent and politeness flag.

Text: {text}



Answer:
""",
        )
    )
    return ChatPromptTemplate.from_messages([system_message, human_message])


augmentation_prompt = get_empathy_prompt()

llm_chain = LLMChain(prompt=augmentation_prompt, llm=llm)

output = llm_chain.run(
    {
        "text": "@Brennan_PB Backs up what I always thought ... yer all as mad as each other",
        "politeness": "impolite",
    }
)

print(output)
