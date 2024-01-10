import csv
import json
import os
from io import StringIO

import requests
from langchain.document_loaders import CSVLoader
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from utils import bedrock, print_ww

boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
)


# Get tcg categories file
def fetch_tcgcsv_file():
    url = "https://tcgcsv.com/Categories.csv"
    tcg_response = requests.get(url)
    return tcg_response.json()


def save_output_to_file(output, filename):
    output_folder = "output-examples"

    # Ensure the output folder exists
    import os

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create and open the file in the output folder
    output_file_path = os.path.join(output_folder, filename)
    with open(output_file_path, "w") as output_file:
        # Write the string content to the file
        output_file.write(output)

    print(f"Output saved to: {output_file_path}")


def retrieve_and_read_csv(csv_url):
    """
    Retrieve a CSV file from a URL and return a CSV reader object
    :param csv_url:
    :return:
    """
    try:
        # Retrieve the CSV file from the URL
        tcgcsv_response = requests.get(csv_url)
        tcgcsv_response.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx)
        csv_content = StringIO(tcgcsv_response.text)
        # Save the CSV content to a file
        with open(
            "data/tcg_categories.csv", "w", newline="", encoding="utf-8"
        ) as csv_file:
            csv_file.write(tcgcsv_response.text)
        retrieved_csv_reader = csv.reader(csv_content)
        return retrieved_csv_reader

    except Exception as e:
        print(f"Error: {e}")
        return None


def save_csv_and_generate_code(url):
    # Example usage:
    csv_reader = retrieve_and_read_csv(url)

    # Create the prompt
    # Analyzing sales

    prompt_data = """

Human: You have a CSV, with columns:
    - categoryId
    - name 
    - modifiedOn 
    - displayName 
    - seoCategoryName 
    - categoryDescription 
    - categoryPageTitle 
    - sealedLabel 
    - nonSealedLabel 
    - conditionGuideUrl 
    - isScannable 
    - popularity 
    - isDirect
    
    Create a python program to analyze the products sold in the CSV. The program should output:
    
    - Name of the category 
    - The ID of the category 
    - Popularity value of the product (number of units sold) 
    - How many categories exist 
    - What the percentage of the proportion of the most popular category overall compared to 
    all the other categories combined
    - The popularity score
    
    Ensure the code is syntactically correct, bug-free, optimized, not span multiple lines unnecessarily, and prefer 
    to use standard libraries. 
    
    Return only python code without any surrounding text, explanation or context. When you 
    return the code, do not wrap it in Markdown backticks or formatting, only respond with the code itself. When you 
    print the code, be sure to indicate the output you are printing, as well as the output of the object printed. 
    Trim all leading whitespace from the first line of the code. Be sure to load data from the following file path 
    ../data/tcg_categories.csv
    
    
    Assistant:
    """

    # Claude - Body Syntex
    body = json.dumps(
        {
            "prompt": prompt_data,
            "max_tokens_to_sample": 4096,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 0.5,
            "stop_sequences": ["\n\nHuman:"],
        }
    )

    modelId = "anthropic.claude-v2"
    accept = "application/json"
    contentType = "application/json"

    response = boto3_bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())
    response_text = response_body.get("completion")
    output_filename = "output.py"
    print_ww(f"Saved file to output/{output_filename}")
    save_output_to_file(response_text, output_filename)


def store_csv_vector_db(document_path):
    br_embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1", client=boto3_bedrock
    )
    # Load CSV document
    csv_loader = CSVLoader(document_path)
    documents = csv_loader.load()
    docs = CharacterTextSplitter(
        chunk_size=2000, chunk_overlap=400, separator=","
    ).split_documents(documents)

    print(f"Number of documents after split and chunking={len(docs)}")
    try:
        vectorstore_faiss_aws = FAISS.from_documents(
            documents=docs, embedding=br_embeddings
        )

        print(
            f"vectorstore_faiss_aws: number of elements in the index={vectorstore_faiss_aws.index.ntotal}:"
        )

    except ValueError as error:
        if "AccessDeniedException" in str(error):
            print(
                f"\x1b[41m{error}\
            \nTo troubeshoot this issue please refer to the following resources.\
             \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
             \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n"
            )

            class StopExecution(ValueError):
                def _render_traceback_(self):
                    pass

            raise StopExecution
        else:
            raise error

    wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss_aws)
    cl_llm = Bedrock(model_id="anthropic.claude-v2", client=boto3_bedrock)
    additional_context = """
    - Name of the category 
    - The ID of the category 
    - Popularity value of the product (number of units sold) 
    - How many categories exist 
    - What the percentage of the proportion of the most popular category overall compared to 
    all the other categories combined
    - The popularity score
    """
    print_ww(
        wrapper_store_faiss.query(
            "What is the popularity score of Magic the gathering compared to Pokemon?"
            f"Be sure to include the following data in a bullet-list format {additional_context}",
            llm=cl_llm,
        )
    )


save_csv_and_generate_code("https://tcgcsv.com/Categories.csv")
store_csv_vector_db("./data/tcg_categories.csv")
