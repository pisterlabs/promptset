from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain  # Import LLMChain
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator
import re

class Course(BaseModel):
    course_code: str
    course_name: str

    @validator('course_code')
    def validate_course_code(cls, v):
        if not re.match(r'^[A-Z]{2,4}\d{2,4}$', v):
            raise ValueError('Invalid course code format')
        return v

    @validator('course_name')
    def validate_course_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Course name cannot be empty')
        return v



if __name__ == "__main__":
    load_dotenv()

    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

    memory = ConversationBufferMemory()

    parser = PydanticOutputParser(pydantic_object=Course)

    long_prompt_txt = """
        You are a CS student. What is a good course code and course name for a class on {topic}? 

        {format_instructions}                   
        """

    prompt = PromptTemplate.from_template(long_prompt_txt, 
        partial_variables={"format_instructions": parser.get_format_instructions()})


    chain = LLMChain(
        llm=chat_model,
        prompt=prompt,
        verbose=True,  # Set verbose to True
        memory=memory
    )

    res = chain({"topic": "software testing"})

    print(res)

    out = parser.invoke(res['text'])

    print(out)


