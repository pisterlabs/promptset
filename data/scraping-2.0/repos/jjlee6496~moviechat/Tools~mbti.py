import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic.v1 import BaseModel, Field
from langchain.agents import tool
from utils import get_openai_api_key

openai.api_key = get_openai_api_key()
model = ChatOpenAI(temperature=0.7, max_tokens= 256)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Your role is to explain about personality given {mbti}\
      Return as detailed as you know about given mbti."),
    ("user", "{mbti}"),
])

# Define the input schema
class MBTI(BaseModel):
    mbti: str = Field(..., description="Extracted mbti from user query")

@tool(args_schema=MBTI)
def get_mbti_explaination(mbti: str) -> str:
    """유저 쿼리로부터 mbti를 판단하고, 추천을 위한 맥락 제공을 위해 mbti에 대한 설명을 반환한다.

    Args:
        mbti (str): 유저 쿼리에서 보이는 유저의 mbti 유형

    Returns:
        mbti_explanation (str): 성격 유형에 따른 설명
    """
    mbti_chain = prompt | model
    
    return mbti_chain.invoke({"mbti": mbti}).content