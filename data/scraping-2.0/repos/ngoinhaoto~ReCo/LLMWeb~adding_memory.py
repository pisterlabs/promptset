from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks import get_openai_callback

from dotenv import load_dotenv

load_dotenv()

def count_tokens(query):
    with get_openai_callback() as cb:
        result = chain.invoke(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful chatbot'),
        MessagesPlaceholder('history'),
        ('human', "{input}")
    ]
)

memory = ConversationBufferMemory(return_messages = True)
print(memory.load_memory_variables({}))

chain = (
    RunnablePassthrough.assign(
        history = RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    | prompt
    | model
)

inputs = {"input": "My interest here is to explore the potential of integrating Large Language Models with external knowledge"}
# response = count_tokens_runnable.invoke(chain, inputs)
response = count_tokens(chain, inputs)
print(response.content)

memory.save_context(inputs, {"output": response.content})

print(memory.load_memory_variables({}))


inputs = {"input": "I just want to analyze the different possibilities. What can you think of?"}
response = count_tokens(chain, inputs)
print(response.content)
memory.save_context(inputs, {"output": response.content})
print(memory.load_memory_variables({}))
