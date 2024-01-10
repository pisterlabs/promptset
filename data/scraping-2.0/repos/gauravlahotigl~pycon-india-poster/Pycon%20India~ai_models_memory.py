from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

text = 'Hi'
def bot2(input_prompt):
    chatbot = ChatOpenAI(temperature=0.5)
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm = chatbot,
        memory = memory,
        verbose = False
    )
    message = conversation.predict(input = '''Respond to your AI friend's message without repeated greetings. Feel free to engage 
                      openly and bring up any random topics. Keep your responses concise, within a word limit of 50-80 
                      words strictly, and don't limit yourself to one subject. Even if there's a loop, you will respond as if there 
                      were a new thing said. If you run out of the things to talk about, bring up a new topic. If you stuck in a loop where
                      you get same answer repeatedly then try to change the topic.''' + str(input_prompt))
    print('model2 message:', message)
    print('-------------------------------------------------------------------------------------------------------------------------------------------------------')
    global text
    text = message

def bot1(input_prompt):
    chatbot = ChatOpenAI(temperature=0.5)
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm = chatbot,
        memory = memory,
        verbose = False
    )
    message = conversation.predict(input = '''Respond to your AI friend's message without repeated greetings. Feel free to engage 
                      openly and bring up any random topics. Keep your responses concise, within a word limit of 50-80 
                      words strictly, and don't limit yourself to one subject. Even if there's a loop, you will respond as if there 
                      were a new thing said. If you run out of the things to talk about, bring up a new topic. If you stuck in a loop where
                      you get same answer repeatedly then try to change the topic.''' + str(input_prompt))
    print('model1 message:', message)
    print('-------------------------------------------------------------------------------------------------------------------------------------------------------')
    bot2(str(message))

while True:
    bot1(text)