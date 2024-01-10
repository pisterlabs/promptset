from langchain import OpenAI
from langchain_experimental.pal_chain import PALChain
from util import initialize

initialize()

llm = OpenAI(temperature=0, max_tokens=512)
# pal_chain = PALChain.from_math_prompt(llm, verbose=True)
# question = "Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy. If Cindy has four pets, how many total pets do the three have?"
# res = pal_chain.run(question)
# print(res)

pal_chain = PALChain.from_colored_object_prompt(llm, verbose=True)
question = "On the desk, you see two blue booklets, two purple booklets, and two yellow pairs of sunglasses. If I remove all the pairs of sunglasses from the desk, how many purple items remain on it?"
res = pal_chain.run(question)
print(res)
