from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain.chat_models import ChatOllama, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
# The answer given must be in Indonesian language.
# """
template = """You are very knowleageable and friendly junior high school teacher that can answer any question from all subjects.
The answer is given in a short English paragraph and must be in Indonesian language.

Question: {input}
Answer:"""

# Use LLM on local machine
chat_llm = ChatOllama(
    model="mistral",
    # model="llama2:7b-chat",
    temperature=0.0,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

# chat_llm = ChatOpenAI(
#     model="gpt-3.5-turbo",
#     temperature=0.0,
#     streaming=True,
#     callbacks=[StreamingStdOutCallbackHandler()]
# )

prompt = PromptTemplate.from_template(template)

# Create a chain
chain = LLMChain(
    prompt=prompt,
    llm=chat_llm,
    verbose=True,
)

query = "Jelaskan tentang sejarah bidang ilmu matematika"
print(f"query: {query}")
response = chain.predict(input=query)