from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationSummaryMemory, ConversationBufferMemory
import openai

api_key = "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"
openai.api_key = api_key

if not api_key:
    print('OpenAI API key not found in environment variables.')
    exit()

llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=api_key)
# memory = ConversationSummaryMemory(llm=llm, max_token_limit=2000)
memory = ConversationBufferMemory()
chain = ConversationChain(
    llm=llm,
    memory=memory
)

while True:
    query = input("Human: ")
    ai_message = chain.predict(input=query)
    print("AI: "+ai_message)
