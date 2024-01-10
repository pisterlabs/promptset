"""Module providing a function to summarize content"""

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import utilities.llm_definitions as llm_definitions

def summerize(objective, content):
    """Generate a summary of the content based on the objective using LLM

    Args:
        objective (str): The query prompt we are asking the LLM to evaluate
        content (str): The documents that we are using in context

    Returns:
        str: The summary of the content based on the objective.
    """
    if len(content) < 10000:
        return content

    llm = llm_definitions.llm_ollama

    # Break the text into small chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=10000,
        chunk_overlap=500
        )
    docs = text_splitter.create_documents([content])
    
    # Generate the Prompt Template that will be used in the chain
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    prompt_template = PromptTemplate(
        template=map_prompt,
        input_variables=["text", "objective"]
        )

    # Create Summary Chain
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=prompt_template,
        combine_prompt=prompt_template,
        verbose=True
        )
    
    # Run the query and return the output
    output = summary_chain.run(input_documents=docs, objective=objective)
    
    return output
