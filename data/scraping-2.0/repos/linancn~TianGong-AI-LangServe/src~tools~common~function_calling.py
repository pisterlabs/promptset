from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough


def function_calling(
    function_desc: str,
    function_para: dict,
    prompt: ChatPromptTemplate,
    model_name: str,
    query: str,
):
    function = {
        "name": "function_calling",
        "description": function_desc,
        "parameters": function_para,
    }
    model = ChatOpenAI(model=model_name, temperature=0).bind(
        function_call={"name": "function_calling"}, functions=[function]
    )
    runnable = {"input": RunnablePassthrough()} | prompt | model

    response = runnable.invoke(query)

    result = response["additional_kwargs"]["function_call"]["arguments"]

    return result
