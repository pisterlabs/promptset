from langchain.chat_models import ChatOpenAI
from configs.model_config import llm_model_dict, LLM_MODEL
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

model = ChatOpenAI(
    streaming=True,
    verbose=True,
    # callbacks=[callback],
    openai_api_key=llm_model_dict[LLM_MODEL]["api_key"],
    openai_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
    model_name=LLM_MODEL
)


human_prompt = "{input}"
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [("human", "Hi"),
     ("ai", "Hello"),
     ("human", "{input}")])


chain = LLMChain(prompt=chat_prompt, llm=model, verbose=True)
print(chain({"input": "How are you"}))