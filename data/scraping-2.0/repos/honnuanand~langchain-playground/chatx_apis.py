from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import LLMChain

from fastapi import APIRouter
from pydantic import BaseModel, Field


#create an router instance
router = APIRouter(
    prefix="/lang",
    tags=["lang"],
    responses={404: {"description": "Not found"}},
)
from pydantic import BaseModel

router = APIRouter()

class TranslationRequest(BaseModel):
    
    input_language: str = Field(..., example="English")
    output_language: str = Field(..., example="Spanish")
    text: str = Field(..., example="Hello, how are you?")
    

class TranslationResponse(BaseModel):
    translated_text: str


def translate_text(input_language, output_language, text):
    chat = ChatOpenAI(temperature=0, openai_api_key="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")

    template = "You are a helpful assistant that accepts conversations in {input_language} and replies back in {output_language} pretending to be a human."
    # print(template)
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    return chain.run(input_language=input_language, output_language=output_language, text=text)


@router.post("/chatx", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    input_language = request.input_language
    output_language = request.output_language
    text = request.text
    translated_text = translate_text(input_language, output_language, text)
    return {"translated_text": translated_text}
