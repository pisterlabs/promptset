#! /usr/bin/env python
from kor.extraction import create_extraction_chain
from kor.nodes import Object
from langchain.chat_models import ChatOpenAI
from datetime import datetime

with open("schema.json", "r") as f:
    schema = Object.parse_raw(f.read())

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k",
    temperature=0,
    max_tokens=2000
)

chain = create_extraction_chain(
    llm, 
    schema, 
    encoder_or_encoder_class="JSON", 
    verbose=False
)

def parse(query: str):
    data = f"""
    Current Date: {datetime.today()}
    Query: {query}
    """
    return chain.run(text=data)['data']