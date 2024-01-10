import dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage, )

# Load OPENAI_API_KEY
dotenv.load_dotenv()

chat = ChatOpenAI()

# __call__
# One message
print(chat([HumanMessage(content="Translate this sentence from English to French: I love programming.")]))

# Multiple messages
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="I love programming.")
]
print(chat(messages))

# generate
batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages)
print(result)
print(result.llm_output)

print("Base Language Model interface")
print(chat.predict("Tell me a joke"))
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="I love programming.")
]
print(chat.predict_messages(messages))
