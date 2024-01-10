from langchain.chains import load_chain 

def calc():
    
    chain = load_chain("lc://chains/llm-math/chain.json")
    chain.run("whats 2 raised to .12")
    