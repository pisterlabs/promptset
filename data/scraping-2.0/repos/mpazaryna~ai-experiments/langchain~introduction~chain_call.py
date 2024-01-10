# chain_call.py
from langchain import ConversationChain, PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# --------------------------------------------------------------
# Chains: Combine LLMs and prompts in multi-step workflows
# --------------------------------------------------------------


def run_chain_demo():
    llm = OpenAI()
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run("AI Chatbots for Podiatry Offices")
