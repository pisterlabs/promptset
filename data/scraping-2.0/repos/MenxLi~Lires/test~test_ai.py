from lires_ai.lmInterface import OpenAIChatStreamIter, streamOutput
from lires_ai import globalConfig as config
import openai

text = ""
print("generating....")

# config.openai_api_base = "https://api.aiproxy.io/v1"
# config.fastchat_api_base = ""
# ai = OpenAIStreamIter(model="vicuna-13b")
# ai = OpenAIChatStreamIter(model="gpt-4-32k")
# ai = OpenAIChatStreamIter(model="gpt-4-32k")
openai.api_key = "_"
openai.api_base = "http://localhost:8000/v1"
# ai = OpenAIChatStreamIter(model="/home/monsoon/Data/llm_weights/yarn-llama-2-13b-64k.Q4_K_M.gguf")
ai = OpenAIChatStreamIter(model="/home/monsoon/Data/llm_weights/ggml-llama-2-13b-chat-q4_k_m.gguf")

streamOutput(ai("Write a story about a robot that is trying to take over the world in around 100 words."), print_callback=print)
streamOutput(ai("Summarize the story in 50 words"), print_callback=print)

print("------ Conversations ------")
print(str(ai.conversations))