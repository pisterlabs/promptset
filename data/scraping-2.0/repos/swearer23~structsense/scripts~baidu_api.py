from dotenv import load_dotenv
from langchain.chat_models import QianfanChatEndpoint 
from langchain.chains import create_extraction_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from pydantic import BaseModel, Field, validator
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models.base import HumanMessage

load_dotenv()

schema = {
    "properties": {
        "name": {"type": "string"},
        "height": {"type": "integer"},
        "hair_color": {"type": "string"},
    },
    "required": ["name", "height"],
}

contract_schema = {
    "properties": {
      "房屋出租方姓名": {"type": "string"},
      "房屋出租方性别": {"type": "string"},
      "房屋出租方身份证号码": {"type": "string"},
      "房屋承租方姓名": {"type": "string"},
      "房屋承租方性别": {"type": "string"},
      "房屋承租方身份证号码": {"type": "string"},
      "出租房屋坐落地址": {"type": "string"},
      "原合同编号": {"type": "string"},
      "本合同编号": {"type": "string"},
      "建筑面积": {"type": "string"},
      "每月租金": {"type": "string"},
    },
    "required": ["房屋出租方", "房屋承租方"],
}

class Contract(BaseModel):
    renter: str = Field(..., title="房屋出租方姓名")
    rentee: str = Field(..., title="房屋承租方姓名")
    renterName: str = Field(..., title="房屋出租方姓名")
    renteeName: str = Field(..., title="房屋承租方姓名")
    renterGender: str = Field(..., title="房屋出租方性别")
    renteeGender: str = Field(..., title="房屋承租方性别")
    renterID: str = Field(..., title="房屋出租方身份证号码")
    renteeID: str = Field(..., title="房屋承租方身份证号码")
    address: str = Field(..., title="出租房屋坐落地址")
    oldContractID: str = Field(..., title="原合同编号")
    newContractID: str = Field(..., title="本合同编号")
    area: str = Field(..., title="建筑面积")
    rent: str = Field(..., title="每月租金")

class POContract(BaseModel):
    phone: str = Field(..., title="电话")
    fax: str = Field(..., title="传真")
    contactName: str = Field(..., title="联系人")
    contactPhone: str = Field(..., title="联系电话")
    contactEmail: str = Field(..., title="联系邮箱")

parser = PydanticOutputParser(pydantic_object=POContract)
prompt = PromptTemplate(
    template='''
    You are a bot to extract assigned content from given content or file.
    Answer the user query based on instruction and query content.
    {format_instructions}\n{query}\n.
    response only json format of extracted content according to schema, do not add anything else.
    the response should starts with \'\'\'json and ends with \'\'\'.
    ''',
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# loader = PyPDFLoader('docs/lianjia.pdf')
loader = PyPDFLoader('docs/PO.PDF')

text_splitter = CharacterTextSplitter(
    chunk_size = 5000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)
pages = loader.load_and_split(text_splitter)
query = '\n'.join([page.page_content for page in pages])

# # Run chain
# chain = create_extraction_chain(contract_schema, llm)
# output = chain.run(text.replace('\n', '\\n').replace('\t', '\\t'))
# print(output)

_input = prompt.format_prompt(query=query)
# print(_input.to_string())
model = QianfanChatEndpoint(temperature=0.1, model="ERNIE-Bot")
output = model([HumanMessage(content=_input.to_string())])
print(output.content)
result = parser.parse(output.content)
print(result)
