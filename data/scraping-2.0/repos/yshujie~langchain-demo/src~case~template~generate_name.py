from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI


def generateName():
    llm = OpenAI(temperature=0.9)
    text = "We would be a good company name for a company that make colorful socks."
    
    return llm(text)

def genNameWithPrompt():
    prompt = PromptTemplate(
        input_variables=["product"],
        template="what is a good name for a company that makes {product}?",
    )
    
    return prompt.format(product="colorful socks")

def genNameWithChain():
    llm = OpenAI(temperature=0.9)
    
    prompt = PromptTemplate(
        input_variables=["product"],
        template="what is a good name for a company that makes {product}?",
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run("colorful socks")