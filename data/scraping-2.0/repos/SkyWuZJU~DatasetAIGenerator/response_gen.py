import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://xiaoma-openai.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "44a43d50221443c690df1c2fd0f9cc14"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chat_models import azure_openai

reponse_gen_prompt = PromptTemplate(
    input_variables=['GivenPrompt'],
    template = "{GivenPrompt}" # 直接使用待生成response的prompt
)

def generate_response(given_prompt):
    final_prompt = reponse_gen_prompt.format(GivenPrompt=given_prompt)
    messages = [
        HumanMessage(content = final_prompt)
    ]
    llm = azure_openai.AzureChatOpenAI(
        deployment_name = 'turbo35',
        temperature = 0,
    )
    response = llm(messages).content
    # 缺少对response的过滤，因为可能有回复失败等现象？
    return response