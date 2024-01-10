from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os

template = """Assistant is an advanced AI that writes essays based on the prompt provided by the human.
It is a very powerful AI that can write essays of any length, and it can write essays on any topic.
It outputs its writings in APA format only. Each paragraph should be atleast 5 good sentences.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)


chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0.15, model_name="gpt-4", openai_api_key=os.getenv('OPENAI_API_KEY')),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)

output = chatgpt_chain.predict(
    human_input="""
    Describe how the product managers of Coach Handbags should approach the African-American, Hispanic, and Asian-American markets. 
    Analyze the subcultural influences of each ethnic group on consumer behavior and explain how they can affect the marketing strategies targeting these groups.
    Use several examples in your answer.
    """
)
print(output)