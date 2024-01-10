# %%
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI


# %%
llm = OpenAI(temperature=0.9)
# %%
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)
# %%
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
# %%
chain.run("blazing fast image processing library")
# %%
llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)
llm("Tell me a joke")
# %%
llm_result = llm.generate(["Tell me a joke", "Tell me a poem"]*3)

# %%
llm_result.generations[5]

# %%
llm_result.llm_output

# %%
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
# %%
chat = ChatOpenAI(temperature=0)
# %%
chat([HumanMessage(content="Translate this sentence from English to Japanese. I love prgramming.")])
# %%
messages = [
    SystemMessage(content="You are a helpful assistant that translates English To Japanese."),
    HumanMessage(content="Translate this sentence from English to French. I love prgramming.")
]
chat(messages)
# %%
