import os
from urllib.parse import urlparse

from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from prompt import QUERY_TO_PYDANTIC_PROMPT

from kintone import Kintone
from dotenv import load_dotenv

from tools.kintone_field_to_model import kintone_field_to_model
from tools.request import get_url_content

load_dotenv()

DOMAIN = os.getenv("KINTONE_DOMAIN")
SPACE_ID = os.getenv("KINTONE_SPACE_ID")
USER = os.getenv("KINTONE_USER")
PASSWORD = os.getenv("KINTONE_PASSWORD")


def client() -> Kintone:
    kintone = Kintone(domain=DOMAIN, user=USER, password=PASSWORD)
    return kintone


@tool
def get_apps() -> list:
    """Returns a list of kintone applications"""
    result = client().apps.get(space_ids=[int(SPACE_ID)])
    return result


@tool
def get_app(app_id: int) -> dict:
    """Return properties of selected kintone application"""
    result = client().form.get(app_id=app_id)
    return result


@tool
def get_records(app_id: int) -> list:
    """Return records of selected kintone application"""
    result = client().records.get(app_id=app_id)
    return result


@tool
def add_record(app_id: int, query: str) -> dict:
    """Add record to selected kintone application. Please check the details of the application you are registering for in advance and make sure that the BODY is set up with all the necessary information to register the record."""
    pydantic_instance = query_to_pydantic_instance(app_id, query)
    result = client().records.create(app_id=app_id, records=[pydantic_instance.dict()])
    return result


def form_schema_to_pydantic_model(app_id: int):
    schema = client().form.get(app_id=app_id)
    return kintone_field_to_model(schema["properties"])


def query_to_pydantic_instance(app_id: int, query: str):
    if is_url(query):
        content = get_url_content(query)
        query = "\n".join([query, content])
    model = form_schema_to_pydantic_model(app_id)
    prompt = ChatPromptTemplate.from_template(QUERY_TO_PYDANTIC_PROMPT)
    llm = ChatOpenAI(model="gpt-4", temperature=0.0)
    parser = PydanticOutputParser(pydantic_object=model)
    chain = prompt | llm | parser
    return chain.invoke(
        {"format_instructions": parser.get_format_instructions(), "query": query}
    )


def is_url(query: str) -> bool:
    try:
        result = urlparse(query)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
