from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate
def load_llm(max_tokens, prompt_template):
    # Load the locally downloaded model here
    llm = CTransformers(
        # model = "llm\llama-2-7b-chat.ggmlv3.q8_0.bin",
        model = "llm\pytorch_model-00001-of-00003.bin",
        model_type="llama",
        max_new_tokens = max_tokens,
        temperature = 0.2
    )
    
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    print(llm_chain)
    return llm_chain