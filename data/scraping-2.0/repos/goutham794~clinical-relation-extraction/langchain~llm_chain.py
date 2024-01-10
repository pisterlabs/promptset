from langchain.llms import OpenAI, AzureOpenAI
from langchain.chat_models import ChatOpenAI



from langchain.chains import LLMChain


def get_llm_chain(prompt, args):
    if args.llm_service == 'openai':
        llm = ChatOpenAI(temperature=0, model="gpt-4")
    else:
        llm = AzureOpenAI(deployment_name="Davinci", model_name="text-davinci-003")

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain
