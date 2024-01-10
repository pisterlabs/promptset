from dotenv import load_dotenv
load_dotenv()

import os
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


from langchain import OpenAI,ConversationChain,LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

template="""Assistant is a large language model trained by OpenAI.

Assistant's job is to answer questions asked by the user.
Assistant must ask clarification questions and must not make answers up if assisitant does not know the answer to any of the questions.

{history}
Human: {human_input}
Assistant:"""

chat = OpenAI(temperature = 0)

prompt = PromptTemplate(
    input_variables = ["history","human_input"],
    template = template
)

chatgpt_chain = LLMChain(
    llm = OpenAI(temperature=0),
    prompt = prompt,
    verbose = True,
    memory = ConversationBufferWindowMemory(k=2),
)

query = input("Hello! How can I help you today?")

chatgpt_chain.predict(human_input= query)