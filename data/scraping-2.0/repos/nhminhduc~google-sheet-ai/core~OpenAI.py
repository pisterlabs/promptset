import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def llm_run(model: str, temperature: float, template: str, **kwargs):
    template_variables = re.findall(r"\{(.*?)\}", template)
    input_variables = {key: (kwargs.get(key) or "") for key in template_variables}
    # print(input_variables)

    llm = ChatOpenAI(model_name=model, temperature=temperature)
    prompt_template = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.run(**input_variables)
    return result
