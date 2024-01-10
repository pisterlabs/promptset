import logging
import time 
import os
import io
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.memory import ConversationTokenBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import Tool
from langchain.chains import LLMChain
from run_chain import get_chain
from shared.selector import get_llm, get_embedding
from documents.document_loader import get_database
from agent_tools.vector_store_retrieval_qa_tool import VectorStoreRetrievalQATool
from utilities.calculate_timing import convert_milliseconds_to_english

def complete_template(template_file_name):

    # Time it
    start_time = time.time()

    # get an output file path
    completed_document_path = os.path.splitext(template_file_name)[0] + "_completed.md"

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

    # For the template completion analysis, use an Agent with access to various tools containing different Remote Control documents
    consolidation_llm = get_llm(False)
    local_llm = get_llm(True)
    #openai_embeddings = get_embedding(False)
    #fda_db = get_database(openai_embeddings, "fda")

    # llm_chain = LLMChain(llm=openai_llm, verbose=True, return_source_documents=True, callbacks=[callback_handlers.DebugCallbackHandler()]

    # open_ai_chain = LLMChain.from.from_llm(llm, db.as_retriever(search_kwargs=vectordbkwargs), memory=memory, verbose=verbose, return_source_documents=True, callbacks=[callback_handlers.DebugCallbackHandler()])


    # Create tools for querying the different databases
    rc_design_and_dev_tool = VectorStoreRetrievalQATool(None, "rc_design_and_dev", True, verbose=True, override_llm=local_llm, return_source_documents=True)
    rc_design_input_tool = VectorStoreRetrievalQATool(None, "rc_design_input", True, verbose=True, override_llm=local_llm, return_source_documents=True)
    rc_design_output_tool = VectorStoreRetrievalQATool(None, "rc_design_output", True, verbose=True, override_llm=local_llm, return_source_documents=True)
    
    # tools = [
    #     Tool("Remote Control Design and Development Process and Plans Tool",
    #          rc_design_and_dev_tool.run,
    #          "Useful for queries about Remote Control that are about Design and Development Process and Plans including:\nClinical Evaluation Plans\nDesign Verification and Validation Test Plans\nGeneral Development Plans\nGlobal Regulatory Strategy Plans\nHuman Factors Plans\nPackaging Plans\nManufacturing Plans\nProduct Security Plans\nSoftware Development Plans\nSoftware Maintenance Plans\nRisk Management Plans\nService Plans\nSupplier Quality Assurance Plans\nInput should be a fully formed question, not referencing any obscure pronouns from the conversation before."),
        
    #     Tool("Remote Control Design Inputs Tool",
    #          rc_design_and_dev_tool.run,
    #          "Useful for queries about Remote Control that are about Design Inputs including:\nHardware Requirements and Specifications\nProduct Requirements\nSoftware Requirements and Specifications\nUser Needs\nUse Specifications\nInput should be a fully formed question, not referencing any obscure pronouns from the conversation before."),

    #     Tool("Remote Control Design Outputs Tool",
    #          rc_design_and_dev_tool.run,
    #          "Useful for queries about Remote Control that are about Design Outputs including:\nArchitecture Documents\nProduction Procedures\nProvisioning Procedures\nServicing Procedures\nSoftware Design Documents\nHardware Design Documents\nSoftware Bill of Materials (SBOMs)\nBinaries Transfer Work Instructions\nBuild and Configuration Management Processes\nSoftware Reference Drawings\nImage Creation Instructions\nInput should be a fully formed question, not referencing any obscure pronouns from the conversation before.")
    # ]

    # Taking memory out of this, because the LLM ends up relying on answers to previous questions
    #memory = ConversationTokenBufferMemory(llm=openai_llm, max_token_limit=4096, memory_key="chat_history", return_messages=True)
    #agent_chain = initialize_agent(tools, openai_llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True)


    output_document = f"# {md_header_splits[1].metadata['Header 1']}\n"

    ## For each of the "headers" in the template, run an LLM to answer the questions!
    for doc in md_header_splits:
        # Write out the headers    
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

        # Split the document section into lines
        text_lines = doc.page_content.split("\n") 

        instruction_lines = []
        guidance:str = ''

        for line in text_lines:
            if line.startswith("[Instructions]"):
                instruction_lines.append(line.strip("[Instructions]"))
            else:
                guidance += "\n" + line + "\n"
        
        # For testing
        #result = {"answer": "something something bla bla"}
        # Initial question
        #result = chain({"question": question, "chat_history": []})
        # Agent - doesn't really work well
        #result = agent_chain.run(chat_history=[], input="Use a tool and answer this query in as much detail as possible: " + question)
        for instruction in instruction_lines:
            design_and_dev_answer = rc_design_and_dev_tool.run("answer this query in as much detail as possible: " + instruction)
            design_input_answer = rc_design_input_tool.run("answer this query in as much detail as possible: " + instruction)
            design_output_answer = rc_design_output_tool.run("answer this query in as much detail as possible: " + instruction)

            other_answers = [
                "Design & Development Tool Asnwer: " + design_and_dev_answer, 
                "Design Inputs Tool Asnwer: " + design_input_answer, 
                "Design Outputs Tool Asnwer: " + design_output_answer
            ]

            other_answers_string = "\n".join(other_answers)

            # Follow-up to make sure the initial question was properly answered
            follow_up_query = f"Given the context provided by the guidance:\n\n{guidance}\n\nInclude the answers given by other tools:\n\n{other_answers_string}\n\nCreate a detailed response to this query: {instruction}"

            result = consolidation_llm.predict(text=follow_up_query + "\n\nHere is a consolidated and well-formatted answer in Markdown (sources listed at the end):\n")

            #result = open_ai_chain({"question": follow_up_query, "chat_history": []})

            output_document += f"*{instruction.strip()}*\n"
            # Chain
            #output_document += f"{result['answer']}\n\n"
            # LLM or Agent
            output_document += f"{result}\n\n"

        output_document += f"\n{guidance}\n\n"

    with io.open(completed_document_path, "w") as file:
        file.write(output_document)


    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.debug("Total Operation Time: ", convert_milliseconds_to_english(elapsed_time * 1000))

    return completed_document_path