from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Instantiate chat model
chat_model = ChatOpenAI(verbose=True)

# Use conversation buffer memory
# chat_memory = ConversationBufferMemory(
#    chat_memory=FileChatMessageHistory("chat_messages.json"),
#    memory_key="chat_messages",
#    return_messages=True)

# Use conversation summary memory with language model same as chat model
chat_memory = ConversationSummaryMemory(
    memory_key="chat_messages",
    return_messages=True,
    llm=chat_model)

# Instantiate chat prompt
chat_prompt = ChatPromptTemplate(
    input_variables=["content", "chat_messages"],
    messages=[
        MessagesPlaceholder(variable_name="chat_messages"),
        HumanMessagePromptTemplate.from_template("{content}")])

# Instantiate chat chain
chat_chain = LLMChain(llm=chat_model, prompt=chat_prompt, memory=chat_memory, verbose=True)

while True:
    content = input(">>> ")
    result = chat_chain({"content": content})
    print(result["text"])
