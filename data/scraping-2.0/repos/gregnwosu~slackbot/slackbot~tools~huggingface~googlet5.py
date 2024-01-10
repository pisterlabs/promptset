from langchain import HuggingFaceHub, LLChain  
# TODO https://huggingface.co/gorilla-llm/gorilla-7b-hf-delta-v0?text=My+name+is+Mariama%2C+my+favorite

hub_llm = HuggingFaceHub(model_name="gorilla-llm/gorilla-7b-hf-delta-v0")
hub_chain = LLChain(llm=hub_llm)