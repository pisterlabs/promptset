from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from packages.models import CHAT_LLM, CHAT_LLM_4
from packages.functions import print_green, print_blue
from langchain.output_parsers import PydanticOutputParser

# TODO: based on input, extract keyword and category


# Define a Pydantic model
class KeywordCategory(BaseModel):
    keyword: list[str] = Field(description="Extract keyword from the input")
    category: str = Field(description="""
    Choose from the following categories:\n

    Categories: ['general', 'keyword', 'content, 'similar', 'trending']
    - 'general': Use this category for questions that are not specifically related to OTT programs.
    - 'content': This category is for questions that seek detailed information about a program, such as its plot, actors, release dates, cast, and characters.
    - 'keyword': Choose this category for inquiries where the user doesn't provide specific program information but asks about actors, release dates, titles, cast, and other keyword-based queries.
    - 'similar': Use this category when users are looking for recommendations similar to a specific program.
    - 'trending': This category is for questions about programs that need viewing data to provide trending information.:\n
    """)

# Set up the models
llm3 = CHAT_LLM
llm4 = CHAT_LLM_4

# Set up the output parser
parser = PydanticOutputParser(pydantic_object=KeywordCategory)

# Set up the prompt template
prompt_template_text = """
Extract keyword and category from the input:\n
{format_instructions}\n
{query}
"""

# Set up the format instructions & prompt template
format_instructions = parser.get_format_instructions()
prompt_template = PromptTemplate(
    template=prompt_template_text,
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

input = [
    "멀티버스가 뭐야?", # general
    "오늘의 날씨", # general
    "쑥의 효능", # general
    "멀티버스", # keyword
    "런닝맨 아이돌 나오는 편 찾아줘", # content
    "하정우가 출연한 영화 알려줘", # keyword
    "해운대는 무슨 내용이야?", # content
    "상견니랑 비슷한 영화 추천해줘", # similar
    "나혼자 산다 출연진 알려줘", # content
    "약한 영웅이랑 비슷한 콘텐츠 추천해줘", # similar
    "현재 인기작 알려줘", # trending
    "웬만해서 그들을 막을 수 없다", # content
    "한번 다녀왔습니다." # content
]

chain3 = prompt_template | llm3 | parser
chain4 = prompt_template | llm4 | parser

for i in range(len(input)):
    response3 = chain3.invoke({"query": input[i]})
    response4 = chain4.invoke({"query": input[i]})
    print_blue(f"Response for GPT3: {response3}")
    print_green(f"Response for GPT4: {response4}")
    print("\n")


