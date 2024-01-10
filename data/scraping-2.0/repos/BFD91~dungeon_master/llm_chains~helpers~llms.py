from langchain import OpenAI
from langchain.llms import OpenAIChat

openai_api_key = "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"

GPT_3_5 = "gpt-3.5-turbo"

GPT_4 = "gpt-4"

gpt_3_5 = OpenAIChat(temperature=0.7, api_key="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", model_name=GPT_3_5)

gpt_4 = OpenAIChat(temperature=0.7, api_key="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", model_name=GPT_4)

#sota_llm = OpenAI(model_name=SOTA_OPENAI_LLM, openai_api_key=openai_api_key)