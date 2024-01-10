
import os

try: 
  from genai.credentials import Credentials
  from genai.schemas import GenerateParams
  from genai.extensions.langchain import LangChainInterface
  from langchain import PromptTemplate
  from langchain.chains import LLMChain, SimpleSequentialChain

except ImportError:
    raise ImportError("Could not import langchain: Please install ibm-generative-ai[langchain] extension.")


"""

 pip install "ibm-generative-ai[langchain]"
 
 """

# make sure you have a .env file under genai root with
api_key = os.getenv("GENAI_KEY", None)
api_url = os.getenv("GENAI_API", None)

# create creds for watsonx.ai models
creds = Credentials(api_key, api_endpoint=api_url)
params = GenerateParams(decoding_method="greedy", min_new_tokens=3, max_new_tokens=300)
#params = GenerateParams(decoding_method="greedy")
llmp = LangChainInterface(model="ibm/granite-13b-sft", params=params, credentials=creds)

# This is a simple example of a zero shot prompt
print(f"******************************************************")
print(f"  ")
print(f"This is a simple zero shot prompt")
print(f"  ")
print(f"******************************************************")
print(f"  ")
print(f"  ")
print(llmp("Explain Mortgage backed securities in one sentence"))




"""
    Implement a Prompt PromptTemplate
"""
print(f"  ")
print(f"  ")
print(f"******************************************************")
print(f"  ")
print(f"This is a sample using a Prompt Template")
print(f"  ")
print(f"******************************************************")
print(f"  ")
print(f"  ")

template = """
You are a mortgage broker with exertise in financial product.
Explain the concept of {concept} in a paragraph.
"""

promptp = PromptTemplate(input_variables=["concept"],template=template,)
#print(promptp)

conceptp = "closing costs"
print(llmp(promptp.format(concept=conceptp)))


"""
    Implement Chains:  In this section I am taking the initial prompt response 
    and then asking the model to update the response.
"""
print(f"******************************************************")
print(f"  ")
print(f"  ")
print(f"This is an example of chaining 2 prompts together to have the second prompt work in the repsonse of the first prompt. We will be using the first Prompt template from above to feed the second Prompt Template, using a Simple Sequence Chain.")
print(f"  ")
print(f"  ")
print(f"******************************************************")
print(f"  ")
print(f"  ")
#print(promptp)
chain = LLMChain(llm=llmp, prompt=promptp)
#print(chain.run(conceptp))

second_prompt = PromptTemplate(
    input_variables=["ml_concept"], 
    template="Turn this financial description {ml_concept} and explain in techncial terms")
chain_two = LLMChain(llm=llmp, prompt=second_prompt)
overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)
explanation = overall_chain.run(conceptp)
print(explanation)



