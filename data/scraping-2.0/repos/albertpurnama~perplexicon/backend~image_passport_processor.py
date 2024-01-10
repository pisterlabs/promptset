import base64
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field, field_validator
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from PIL import Image
import pytesseract

from openai import OpenAI
client = OpenAI()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "./.seed-knowledge/sample-usa-passport-cropped.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

# Define your desired data structure.
class Passport(BaseModel):
    number: str = Field(description="6-9 digit alphanumeric passport number")
    dob: str = Field(description="date of birth in YYYY-MM-DD format")
    issue_date: str = Field(description="date of issueance in YYYY-MM-DD format")
    expiration_date: str = Field(description="date of expiration in YYYY-MM-DD format")

    @field_validator('number')
    @classmethod
    def name_must_contain_space(cls, v: str) -> str:
        if len(v) < 6 or len(v) > 9:
            raise ValueError("Passport number must be 6-9 digits")
        if not v.isalnum():
            raise ValueError("Passport number must be alphanumeric")
        return v
    
model = ChatOpenAI(temperature=0)

parser = PydanticOutputParser(pydantic_object=Passport) # type: ignore

prompt = PromptTemplate(
    template="Provide full information of the passport from an OCR machine.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


# Load an image
image = Image.open('./.seed-knowledge/passport-max-exposure.png')

# Perform OCR
text = pytesseract.image_to_string(image)

chain = prompt | model | parser 
output:Passport = chain.invoke({"query": text })

