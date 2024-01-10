# https://python.langchain.com/en/latest/modules/chains/examples/llm_math.html


from langchain import OpenAI, LLMMathChain


llm = OpenAI(temperature=0, openai_api_key= "sk-your-key-here")
llm_math = LLMMathChain.from_llm(llm, verbose=True)

llm_math.run("What is 13 raised to the .3432 power?")