# from langchain.chains import LLMChain
# from langchain.globals import set_debug
# from langchain.prompts import PromptTemplate

# from frontend_developer_agent.llms.llama_cpp_api import LlamaCppApi

# set_debug(True)


# def test_llama_cpp_api():
#     MODEL_URL = "http://localhost:8080"
#     TEMPLATE = """You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n\
#     ### Instruction:
#     {instruction}
#     ### Response:
#     {primer}"""

#     prompt = PromptTemplate(
#         template=TEMPLATE,
#         input_variables=["instruction", "primer"],
#     )
#     llm = LlamaCppApi(base_url=MODEL_URL)
#     llm_chain = LLMChain(prompt=prompt, llm=llm)
#     llm_chain({
#         "instruction": "Write an Ionic HTML page with two input fields and a button",
#         "primer": "<!DOCTYPE html",
#     })
