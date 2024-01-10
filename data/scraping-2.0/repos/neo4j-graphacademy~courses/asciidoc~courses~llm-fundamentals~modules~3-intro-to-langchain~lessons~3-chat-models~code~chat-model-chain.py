from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

chat_llm = ChatOpenAI(
    openai_api_key="sk-..."
)

prompt = PromptTemplate(template="""You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.

Question: {question}
""", input_variables=["question"])

chat_chain = LLMChain(llm=chat_llm, prompt=prompt)

response = chat_chain.run(
    question="What are the conditions like?"
)

print(response)
