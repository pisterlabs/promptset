from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

# Get your HuggingFace token from https://huggingface.co/settings/token
huggingfacehub_api_token = ''

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature": 0.6, "max_new_tokens": 500})

template = """You are an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers 
to the user's questions.

{question}
"""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "what is the difference between nuclear fusion and nuclear fission?"

print(llm_chain.run(question))