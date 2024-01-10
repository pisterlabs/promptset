import time
import langchain
from langchain.cache import InMemoryCache
from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage

langchain.llm_cache = InMemoryCache()

chat = BedrockChat(
    model_id="anthropic.claude-v2"
)

start = time.time()
result = chat([
    HumanMessage(content="こんにちは！")
])
end = time.time()
print(result.content)
print(f"実行時間: {end - start}秒")

start = time.time()
result = chat([
    HumanMessage(content="こんにちは！")
])
end = time.time()
print(result.content)
print(f"実行時間: {end - start}秒")