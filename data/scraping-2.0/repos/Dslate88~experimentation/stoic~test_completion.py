from langchain.memory import ConversationBufferWindowMemory

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
conversation_with_summary = ConversationChain(
    llm=OpenAI(temperature=0),
    # We set a low k=2, to only keep the last 2 interactions in memory
    memory=ConversationBufferWindowMemory(k=2),
    verbose=True
)
print(OpenAI(temperature=0))
conversation_with_summary.predict(input="Hi, what's up?")
conversation_with_summary.predict(input="tell me a random fact")
