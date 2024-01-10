from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from configs.model_config import llm_model_dict, LLM_MODEL
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

if LLM_MODEL == "Azure-OpenAI":
    model = AzureChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
        openai_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
        openai_api_version=llm_model_dict[LLM_MODEL]["api_version"],
        deployment_name=llm_model_dict[LLM_MODEL]["deployment_name"],
        openai_api_key=llm_model_dict[LLM_MODEL]["api_key"],
        openai_api_type="azure",
    )
else:
    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
        openai_api_key=llm_model_dict[LLM_MODEL]["api_key"],
        openai_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
        model_name=LLM_MODEL
    )

human_prompt = "{input}"
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [("human", "我们来玩成语接龙，我先来，生龙活虎"),
     ("ai", "虎头虎脑"),
     ("human", "{input}")])

chain = LLMChain(prompt=chat_prompt, llm=model, verbose=True)
print(chain({"input": "恼羞成怒"}))
