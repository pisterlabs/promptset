import logging
import time
import os
import io
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from run_chain import get_chain
from shared.selector import get_llm, get_embedding
from documents.document_loader import get_database
from agent_tools.vector_store_retrieval_qa_tool import VectorStoreRetrievalQATool
from langchain.memory import ConversationTokenBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import Tool
from utilities.calculate_timing import convert_milliseconds_to_english

def ingest_template(template_file_name):
    # Time it
    start_time = time.time()

    # get an output file path
    analyzed_template_path = os.path.splitext(template_file_name)[0] + "_analyzed.md"

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    with io.open(template_file_name, "r") as file:
        markdown_document = file.read()

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    logging.debug(md_header_splits)

    # For the FDA analysis, use OpenAI
    llm = get_llm(False)
    embeddings = get_embedding(False)

    fda_tool = VectorStoreRetrievalQATool(None, "fda", False, verbose=True, override_llm=llm, return_source_documents=True, search_type="similarity", top_k=5)
    iso_tool = VectorStoreRetrievalQATool(None, "iso", False, verbose=True, override_llm=llm, return_source_documents=True, search_type="similarity", top_k=2)
    
    # tools = [
    #     Tool("FDA Documentation Tool",
    #          fda_tool.run,
    #          "Useful for querying FDA documents, such as regulatory and guidance documents. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before. Always include source documents from this tool in your answer."),
        
    #     Tool("ISO/AAMI/IEC Standards Documentation Tool",
    #          iso_tool.run,
    #          "Useful for querying ISO, AAMI, or IEC Standards documents. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before. Always include source documents from this tool in your answer."),
    # ]

    # Taking memory out of this, because the LLM ends up relying on answers to previous questions
    # memory = ConversationTokenBufferMemory(llm=openai_llm, max_token_limit=4096, memory_key="chat_history", return_messages=True)
    # agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    output_document = f"# {md_header_splits[0].metadata['Header 1']}\n"

    ## For each of the "headers" in the template, run an LLM to answer the questions!
    for doc in md_header_splits:
        # Split into lines    
        text_lines = doc.page_content.split("\n")
        
        headers = []
        # Pull out the relevant headers, ignoring the top-level header (# Template)
        for key in doc.metadata.keys():
            if key != "Header 1":
                headers.append(doc.metadata[key])

        # Update the output document
        if "Header 4" in doc.metadata:
            output_document += f"#### {doc.metadata['Header 4']}\n"
        elif "Header 3" in doc.metadata:
            output_document += f"### {doc.metadata['Header 3']}\n"
        elif "Header 2" in doc.metadata:
            output_document += f"## {doc.metadata['Header 2']}\n"
        elif "Header 1" in doc.metadata:
            output_document += f"# {doc.metadata['Header 1']}\n"


        ## For each line in this section, get the instructions, get the refinement prompt
        # I am positive there's a better way to do this
        instructions = []
        refinements = []
        for line in text_lines:
            # Get the instructions 
            # I don't really need to pull these here, but maybe we will combine this process with the document creation later and can use them
            if line.startswith("[Instructions]"):
                instructions.append(line.strip())

            # Get the refinements
            if line.startswith("[Refinement]"):
                refinements.append(line.strip("[Refinement]").strip())    

        refinement_answers = []
        for refinement in refinements:
            # Call the LLMs to answer the refinement query
            refinement_query = "Answer the following query, and phrase it in the terms of guidance that is intended to help someone complete a section of a document.  Include all relevant sources.  Query:\n" + refinement
            fda_answer = fda_tool.run(refinement_query)
            iso_answer = iso_tool.run(refinement_query)
            follow_up_query = f"Using these answers\n\nFDA Answer: {fda_answer}\n\nISO/AAMI/IEC Answer: {iso_answer}\n\n"
            result = llm.predict(text=follow_up_query + "\n\nHere is a consolidated and well-formatted answer in Markdown (sources listed at the end):\n")
            refinement_answers.append(result)

        # Simply reprint the instructions for now
        for instruction in instructions:
            output_document += f"{instruction}\n\n"

        for refinement_answer in refinement_answers:
            output_document += f"**[Regulatory Guidance]:**\n{refinement_answer.strip()}\n\n"    

    with io.open(analyzed_template_path, "w") as file:
        file.write(output_document)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.debug("Total Operation Time: ", convert_milliseconds_to_english(elapsed_time * 1000))

    return analyzed_template_path