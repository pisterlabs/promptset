from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

chat_llm = ChatOpenAI(
    openai_api_key="sk-..."
)

prompt = PromptTemplate(template="""You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.

Context: {context}
Question: {question}
""", input_variables=["context", "question"])

chat_chain = LLMChain(llm=chat_llm, prompt=prompt)

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
