from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
import dotenv

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

memory = ConversationBufferMemory()

llm = OpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory,
)

print(conversation.predict(input="Hi there! I'm Kim"))

print(conversation.predict(input="What's my name?"))
