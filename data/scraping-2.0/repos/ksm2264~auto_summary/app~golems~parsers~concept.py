from typing import List
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from app.golems.models.concept import ConceptNameList, Concept


llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")

list_parser = PydanticOutputParser(pydantic_object=ConceptNameList)
prompt = PromptTemplate(
    input_variables = ["text_chunk"],
    template = '''
    Here is a chunk of text from a scientific paper: {text_chunk}
    Extract different important concepts.
    format like this:{format_instructions}
    ''',
    partial_variables={"format_instructions": list_parser.get_format_instructions()}
)
list_generating_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
list_fixing_parser = OutputFixingParser.from_llm(llm = llm, parser = list_parser)

def get_concept_name_list(chunk: str) -> ConceptNameList:

    concept_list = list_generating_chain.predict(
        text_chunk = chunk
    )

    fixed_list = list_fixing_parser.parse(
        concept_list
    )

    print(fixed_list)

    return fixed_list



concept_parser = PydanticOutputParser(pydantic_object=Concept)
prompt = PromptTemplate(
    input_variables = ["text_chunk", "concept", "chunk_buffer"],
    template = '''
    Here is a chunk of text from a scientific paper: {text_chunk}
    This is the concept you need to either update or leave the same: {concept}
    Do this by examining the existing summary, and adding an updated summary of this concept based on the new text.
    Make sure to be extremely concise with your additions.
    Here is some previous chunks: {chunk_buffer}
    format like this:{format_instructions}
    ''',
    partial_variables={"format_instructions": concept_parser.get_format_instructions()}
)
concept_generating_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
concept_fixing_parser = OutputFixingParser.from_llm(llm = llm, parser = concept_parser)


def get_concept(chunk: str, chunk_buffer: List[str], concept: Concept) -> Concept:

    concept = concept_generating_chain.predict(
        text_chunk = chunk,
        chunk_buffer = chunk_buffer,
        concept = concept
    )

    fixed_concept = concept_fixing_parser.parse(concept)

    print(fixed_concept)

    return fixed_concept