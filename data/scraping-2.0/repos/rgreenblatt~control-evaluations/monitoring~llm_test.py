# %%
from rrutils.llm_api.llm import ModelAPI
from rrutils.llm_api.openai_llm import OpenAIChatModel

model = ModelAPI()
await model(["gpt-3.5-turbo"], [{"role": "user", "content": "Hello, how are you?"}], max_tokens=10)
# %%
