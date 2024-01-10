from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, PromptTemplate
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import PyPDFDirectoryLoader
import glob
import argparse
import os


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)


def custom_summary(pdf_folder, custom_prompt):

    """
    Batch summarize all PDFs in a folder using a custom prompt.
    """
    summaries = []
    for pdf_file in glob.glob(pdf_folder + "/*.pdf"):
        loader = PyPDFLoader(pdf_file)
        docs = loader.load_and_split()
        prompt_template = custom_prompt + """

        {text}

        SUMMARY:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                    map_prompt=PROMPT, combine_prompt=PROMPT)
        summary_output = chain({"input_documents": docs},return_only_outputs=True)["output_text"]
        summaries.append(summary_output)
        with open("ouput.txt", "w") as f:
            for summary in summaries:
                f.write(summary + "\n")   
        
    return summaries


# create an python cli argument parser that asks the user if they want to use summary function or querying



def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Python CLI for custom_summary and flexible_querying")

    # Add the -f argument for specifying the string
    parser.add_argument('-f', '--string', choices=['custom_summary', 'flexible_querying'], required=True,
                        help='Specify the string: "custom_summary" or "flexible_querying"')
    # Add the -d arugment to specify the directory of the pdfs
    parser.add_argument('-d', '--directory', required=True, help='Input directory of the pdfs')

    # Add the -p argument for specifying the file path of the prompt
    parser.add_argument('-p', '--path', required=True, help='Input file path of the prompt')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Print the chosen string and file path
    chosen_string = args.string
    prompt_file_path = args.path
    pdf_directory = args.directory

    print(f'Chosen functionality: {chosen_string}')

    # load prompt from the .txt filepath
    with open(prompt_file_path, "r") as f:
        custom_prompt = f.read()
    
    if chosen_string == "custom_summary":
        results = custom_summary(pdf_directory, custom_prompt)
        print(results)
    elif chosen_string == "flexible_querying":
        loader = PyPDFDirectoryLoader(pdf_directory)
        docs = loader.load()
        
        # Create the vector store index
        index = VectorstoreIndexCreator().from_loaders([loader])

        results = index.query(custom_prompt)
        print(results)

        with open("ouput.txt", "w") as f:
            f.write(results + "\n")  

if __name__ == "__main__":
    main()



