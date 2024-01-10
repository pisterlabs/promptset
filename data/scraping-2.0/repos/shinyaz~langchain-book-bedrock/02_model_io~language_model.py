from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage

chat = BedrockChat(
    model_id="anthropic.claude-v2"
)

result = chat(
    [
        HumanMessage(content="こんにちは！"),
    ]
)

print(result.content)
