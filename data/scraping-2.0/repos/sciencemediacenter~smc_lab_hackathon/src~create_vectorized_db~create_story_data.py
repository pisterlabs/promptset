
import os
import json
import argparse
import chromadb
import logging
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from langchain.document_loaders import TextLoader
from typing import Any, Dict, List
from graphql.request_gql import get_general_query
from dotenv import load_dotenv

load_dotenv()

DATA_LOCATION = os.environ.get("DATA_LOCATION")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def _get_story_data_from_data_collection() -> List[Dict[str, Any]]:
    """Get story data from data collection."""
    story_data = get_general_query(
        table_name="story_meta",
        schema_name="smc",
        return_nodes="""
            story_no, 
            title, 
            type, 
            url, 
            ressort, 
            publication_date, 
            expert_statements {
                expert_affiliation, 
                expert_name, 
                statement,
                question
            },
            smc_content {
                teaser
            }""",
        where_clause='type: {_nin: ["Press Briefing", "Data Report"]}',
        args_clause="order_by: {publication_date: desc}"
    )
    return story_data

def _process_story_data(story_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ Create txt files for each story teaser and story statement. """
    story_metadata: List[Dict[str, Any]] = []
    for story in story_data:
        logging.debug(f"[*] Processing Story: {story['story_no']}")

        # keep only stories with expert statements and teaser
        if len(story["expert_statements"]) == 0 or story["smc_content"][0]["teaser"] == "":
            logging.debug(f"[-] Story {story['story_no']} has no expert statements or teaser. Skipping.")
            continue
        
        # create txt files for each story teaser and story statement   
        story_teaser = f'{story["title"]}\n\n{story["smc_content"][0]["teaser"]}'
        with open(f"{DATA_LOCATION}/story_teaser/{story['story_no']}.txt", "w") as f:
            f.write(story_teaser)

        statements_metadata: List[Dict[str, Any]] = []
        expert_statements = story["expert_statements"]

        if story["type"] == "Science Response": # statement_block for each answer by expert and question -> group by expert
            statements_grouped_by_expert: Dict[str, Dict[str, str]] = {}
            for statement_block in expert_statements:
                expert_name = statement_block["expert_name"]
                question = statement_block["question"]
                question = question if question != None else ""
                if expert_name not in statements_grouped_by_expert.keys():
                    statements_grouped_by_expert[expert_name] = {
                        "statement": f"{question}\n\n{statement_block['statement']}",
                        "expert_name": expert_name,
                        "expert_affiliation": statement_block["expert_affiliation"],
                    }
                else:
                    new_statement = f"{question}\n\n{statement_block['statement']}"
                    current_statements = statements_grouped_by_expert[expert_name]["statement"]
                    statements_grouped_by_expert[expert_name]["statement"] = f"{current_statements}\n\n{new_statement}"

            expert_statements = list(statements_grouped_by_expert.values())

        for i, statement_block in enumerate(expert_statements):
            expert_statement_parts = statement_block["statement"].split("\n")
            expert_statement_parts = [p.strip('" "„“"”" "') for p in expert_statement_parts if (p != "" and p != " ")]
            
            expert_statement = "\n\n".join(expert_statement_parts)            
            story_statement = f'{story["title"]}\n\n{expert_statement}'

            statements_metadata.append({
                "statement_no": f"{story['story_no']}_{i}",
                "expert_name": statement_block["expert_name"],
                "expert_affiliation": statement_block["expert_affiliation"],
            })

            with open(f"{DATA_LOCATION}/story_statements/{story['story_no']}_{i}.txt", "w") as f:
                f.write(story_statement)
        
        # Append story metadata 
        metadata = {
            "story_no": story["story_no"],
            "type": story["type"],
            "url": story["url"],
            "ressort": story["ressort"],
            "publication_date": story["publication_date"],
            "statements_metadata": statements_metadata
        }
        story_metadata.append(metadata)

    return story_metadata

def _fill_teaser_collection(
    metadata: Dict[str, Any], 
    teaser_collection: chromadb.Collection,
    text_splitter: RecursiveCharacterTextSplitter):
    story_no = metadata["story_no"]
    # load the document and split it into chunks
    teaser_loader = TextLoader(f"{DATA_LOCATION}/story_teaser/{story_no}.txt")
    teaser_document = teaser_loader.load()

    teaser_docs = text_splitter.split_documents(teaser_document)

    teaser_metadata = metadata.copy()
    del teaser_metadata["statements_metadata"]
    for i, doc in enumerate(teaser_docs):
        _id = f"{story_no}_{i}_teaser"
        doc_metadata = {**doc.metadata, **teaser_metadata}
        logging.debug("[*] filling teaser collection with story_no", story_no)
        logging.debug("[*] metadata: ", doc_metadata)
        # logging.info("    page_content: ", doc.page_content)

        teaser_collection.add(ids=_id, metadatas=doc_metadata, documents=doc.page_content)

def _fill_statement_collection(
    metadata: Dict[str, Any], 
    statement_collection: chromadb.Collection, 
    text_splitter: RecursiveCharacterTextSplitter
):
    statements_metadata = metadata["statements_metadata"]
    
    for statement in statements_metadata:
        statement_no = statement["statement_no"]
        # load the document and split it into chunks
        statement_loader = TextLoader(f"{DATA_LOCATION}/story_statements/{statement_no}.txt")
        statement_document = statement_loader.load()

        statement_docs = text_splitter.split_documents(statement_document)

        statement_metadata = metadata.copy()
        del statement_metadata["statements_metadata"]
        statement_metadata = {**statement_metadata, **statement}

        for i, doc in enumerate(statement_docs):
            _id = f"{statement_no}_{i}_statement"
            doc_metadata = {**doc.metadata, **statement_metadata, "statement_no": _id}
            logging.debug("[*] filling statement_collection with statement_no", _id)
            logging.debug("[*] metadata: ", doc_metadata)
            # load it into Chroma and export it to a file

            statement_collection.add(ids=_id, metadatas=doc_metadata, documents=doc.page_content)

def _create_vectorized_db(persistent_client: chromadb.PersistentClient):
    """ Create vectorized database (ChromaDB). """

    # ****
    # Define embedding function
    # ****
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-ada-002"
    )

    # ****
    # Create ChromaDB Collections
    # ****
    teaser_collection = persistent_client.get_or_create_collection("story_teaser", embedding_function=embedding_function)
    statement_collection = persistent_client.get_or_create_collection("story_statement", embedding_function=embedding_function)

    # ****
    # Create Text Splitter
    # ****  
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, # characters
        chunk_overlap  = 20,
        length_function = len,
        add_start_index = True,
        # separators=["\n\n\n\n", "\n\n", " ", ""]
    )


    # ****
    # load story metadata
    # ****
    with open(f"{DATA_LOCATION}/story_metadata.json", "r") as f:
        story_metadata = json.load(f)

    # ****
    # Fill ChromaDB
    # ****
    logging.info("[+] Filling ChromaDB - This can take some time.")
    for metadata in story_metadata:
        story_no = metadata["story_no"]
        try:
            _fill_teaser_collection(metadata, teaser_collection, text_splitter)
            _fill_statement_collection(metadata, statement_collection, text_splitter)
        except Exception as e:
            logging.info(f"[-] Error in story no: {metadata['story_no']} ")
            logging.error(str(e))
            continue

if __name__ == "__main__":

    # ****
    # Parse arguments
    # ****
    parser = argparse.ArgumentParser(description="Create vectorized database (ChromaDB). The Data in the database will be the same as the zip file from the readme. You can use this script if you need more control over the creation / vectorization process.")
    parser.add_argument("-k", "--keep", action="store_true", help="Keep temporary files, like the text files for the statements and teaser.")
    parser.add_argument("-j", "--json", action="store_true", help="Keep json file with story metadata.")
    parser.add_argument("-v", "--verbose", action="store_const", help="Debug logging.", dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel, format="%(message)s")


    # ****
    # Get the Story Data from the Data Collection
    # ****
    logging.info("[+] Create story data")
    story_data = _get_story_data_from_data_collection()
    logging.info(f"[+] Got {len(story_data)} stories")

    # ****
    # Process the Story Data
    # ****
    logging.info("[+] Process story data")
    story_metadata = _process_story_data(story_data)


    # Save the Story Metadata
    logging.info("[+] Save story metadata to json file")
    with open(f"{DATA_LOCATION}/story_metadata.json", "w") as f:
        json.dump(story_metadata, f, indent=4)

    # Preparing ChromaDB
    logging.info("[+] Preparing ChromaDB")

    # Check if ChromaDB already exists. If so, delete it.
    if os.path.exists(f"{DATA_LOCATION}/chroma_db"):
        logging.info("[!] ChromaDB already exists. Deleting it.")
        os.system(f"rm -r {DATA_LOCATION}/chroma_db")

    persistent_client = chromadb.PersistentClient(
        path=f"{DATA_LOCATION}/chroma_db", 
        settings=Settings(allow_reset=True)
    ) # should be created once and passed around
    _create_vectorized_db(persistent_client=persistent_client)

    # ****
    # Clean up
    # ****
    if not args.keep:
        logging.info("[+] Remove temporary files")
        os.system(f"rm -r {DATA_LOCATION}/story_teaser/*.txt")
        os.system(f"rm -r {DATA_LOCATION}/story_statements/*.txt")
    
    if not args.json:
        logging.info("[+] Remove story metadata json file")
        os.system(f"rm -r {DATA_LOCATION}/story_metadata.json")


