from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable.utils import ConfigurableField
from langchain.runnables.hub import HubRunnable
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(temperature = 0).configurable_fields(
    temperature = ConfigurableField(
        id = "llm_temperature",
        name = "LLM Temperature",
        description = "The temperature of the LLM model",
    )
)

# print(model.invoke("Pick a random number"))

# model.with_config(configurable = {"llm_temperature": 0.9})\
#     .invoke("Pick a random number")

# prompt = PromptTemplate(
#     input_variables = ['x'],
#     template = "Pick a random number above {x}"
# )

prompt = HubRunnable("rlm/rag-prompt").configurable_fields(
    owner_repo_commit = ConfigurableField(
        id = "hub_commit",
        name = "Hub Commit",
        description = "The Hub commit to pull from"
    )
)

prompt.invoke({"question": "foo", "context": "bar"})

# chain = prompt | model
# # print(chain.invoke({'x' : 0}))
# print(chain.with_config(configurable = {"llm_temperature": 0.9}) \
#       .invoke({'x': 0}))


