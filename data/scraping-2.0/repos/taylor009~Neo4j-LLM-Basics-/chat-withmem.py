import os
from dotenv import load_dotenv
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import LLMChain
load_dotenv()

key = os.getenv("OPENAI_KEY")

chat_llm = ChatOpenAI(
    openai_api_key=key
)

prompt = PromptTemplate(template="""You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.

Context: {context}
Question: {question}
""", input_variables=["context", "question"])

memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)

chat_chain = LLMChain(llm=chat_llm, prompt=prompt, memory=memory)

current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Polzeath", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

response = chat_chain.run(
    context=current_weather,
    question="What is the weather like on Watergate Bay?"
)

print(response)