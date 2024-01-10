import os
import json
import datetime

from langchain.document_loaders import PyPDFLoader
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

#langchain configuration
response_schemas = [
    ResponseSchema(name="title", description="the title as shown in the paper"),
    ResponseSchema(name="abstract", description="the abstract as shown in the paper"),
    ResponseSchema(name="url", description="the url of the paper. If not available, use the identifier of the paper, e.g. arXiv ID, as a google search query. For example, if the DOI is 10.1234/5678, then the url is https://www.google.com/search?q=10.1234/5678"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(
    "Your job is to extract data given the first page of a research paper. If unsure for any field, leave it blank, with the exception being the abstract (if not available, give a discription of the paper)\n{format_instructions}\n{paper_data}")]
    ,
    input_variables=["paper_data"],
    partial_variables={"format_instructions": format_instructions}
)

chat_model = ChatOpenAI(temperature=0)

embeddings_model = OpenAIEmbeddings()

def process_pdf(pdf_path):
    """
    Extracts the title, abstract, and authors from a PDF file.
    """

    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    _input = prompt.format_prompt(paper_data=pages[0])
    output = chat_model(_input.to_messages())
    paper_data = output_parser.parse(output.content)

    #Add embeddings
    paper_data["abstract_embedding"] = embeddings_model.embed_query(paper_data["abstract"])

    return paper_data

def main():
    pdfs_directory = "papers"
    output_file = "papers_data.json"
    papers_data = []
    # processed_files = set()

    #check which filenames have already been written to the output file
    if os.path.exists(output_file):
        with open(output_file, 'r') as json_file:
            papers_data = json.load(json_file)

        processed_files = set([paper_data["file_name"] for paper_data in papers_data])

    # #for each paper in the json, add the embeddings
    # for paper_data in papers_data:
    #     paper_data["abstract_embedding"] = embeddings_model.embed_query(paper_data["abstract"])

    # with open(output_file, 'w') as json_file:
    #     json.dump(papers_data, json_file, indent=2)
    # return

    # Iterate through all PDF files in the specified directory
    for index, file_name in enumerate(os.listdir(pdfs_directory)):
        if file_name.endswith(".pdf") and file_name not in processed_files:

            print(f"Extracting paper number {index + 1} -- {file_name}")
            pdf_path = os.path.join(pdfs_directory, file_name)

            try:
                paper_data = process_pdf(pdf_path)
                print("Finished")

                paper_data["file_name"] = file_name
                paper_data["date_extracted"] = datetime.datetime.now().isoformat()
                papers_data.append(paper_data)
            except Exception as e:
                print(f"Error processing PDF file ${file_name}: ${e}")
                continue

    #Save the data as a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(papers_data, json_file, indent=2)

if __name__ == "__main__":
    main()