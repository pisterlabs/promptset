from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory

# LLM
chat_llm = ChatOpenAI(
    # model="text-davinci-002",
    model="gpt-3.5-turbo",
    verbose=True,
    temperature=0.0
)

# Prompt 
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a friendly chatbot having a conversation with human."
        ),
        # The 'variable_name' is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

memory = ConversationSummaryMemory(
    llm=chat_llm,
    memory_key="chat_history",
    return_messages=True
)

# Chain
chain = LLMChain(
    prompt=prompt,
    llm=chat_llm,
    memory=memory,
    verbose=True
)

print(chain({"question": "Halo ini dengan Ghifary"}))

