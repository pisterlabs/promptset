from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from .models import AssistantAnswer

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

def predict(system, request):
    messages = [SystemMessage(content=system), HumanMessage(content=request)]
    response = llm.predict_messages(messages)
    answer = AssistantAnswer(
        system=system,
        request=request,
        response=response.content
    )
    answer.save()
    return answer