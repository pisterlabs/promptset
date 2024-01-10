# from langchain import PromptTemplate
# import os 
# # os.environ["ALL_PROXY"]="127.0.0.1:11137"
# os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"
# template="""Question: {question}

# Answer: """
# prompt= PromptTemplate(
#         template=template,
#     input_variables=['question']
# )
# # 注意 variables的s
# question="Which team won the figure skating in the Beijing Olympics?"
# from langchain import HuggingFaceHub,LLMChain
# # from .autonotebook import tqdm as notebook_tqdm
# # initialize the Hub
# hub_llm=HuggingFaceHub(
#     repo_id="google/flan-t5-xl",
#     model_kwargs={'temperature':1e-10}
# )
# # create prompt template >LLM chain
# llm_chain=LLMChain(
#     prompt=prompt,
#     llm=hub_llm
# )

# print(llm_chain.run(question))

from langchain import HuggingFaceHub, LLMChain
import os
from langchain import PromptTemplate
os.environ["ALL_PROXY"]="127.0.0.1:11137"
os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"
template = """Question: {question}

Answer: """
prompt = PromptTemplate(
        template=template,
    input_variables=['question']
)

# user question
question = "Which team won the figure skating in the Beijing Olympics?"# 不会回答，妈的
question = "Which NFL team won the Super Bowl in the 2010 season?"# 会回答

# os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# initialize Hub LLM
hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-xl',
    model_kwargs={'temperature':1e-10}
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# ask the user question about NFL 2010
print(llm_chain.run(question))