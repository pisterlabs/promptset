from dotenv import load_dotenv
import os
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import pandas as pd
from scripts import url_parse

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


def run_marketing_report(company_path):
    def get_company_page(company_path):
        url = company_path
        if not url.startswith("https://"):
            url = f"https://{url}"

        print(url)

        loader = UnstructuredURLLoader(urls=[url])
        return loader.load()

    data = get_company_page(company_path=company_path)

    company_name = url_parse.extract_domain_name(company_path)

    print(f"You have {len(data)} document(s)")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)

    docs = text_splitter.split_documents(data)

    print(f"You now have {len(docs)} documents")

    map_prompt = """Below is a section of a website about {prospect}

    Write a concise summary about {prospect}. If the information is not about {prospect}, exclude it from your summary. The purpose of this report is sales prospecting for internal use.

    {text}

    CONCISE SUMMARY:"""
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "prospect"]
    )

    combine_prompt = """
    Your goal is to write a profile report about {prospect} for the use of your sales team .

    Be sure to include important details for the sales process. This should be addressed as an internal report directly to the sales team. Look for ways to leverage the knowledge for sales. You know that your team knows what they are doing so you are breif and to the point. You dont waste time telling them things they already know. You just report the facts.


    INFORMATION ABOUT {prospect}:
    {text}

    INCLUDE THE FOLLOWING PIECES IN YOUR RESPONSE:
    - Gather any contact information you can find. Including specific names, emails and numbers you can collect
    - A breif on the company including where its located, how long its been in operation, size of the business and anyother relevant information
    - Check if social media is available to view and what types. 
    - What other software they may be using on their website including forms, and hosting.
    - Do not make up any data that is not directly referenced in the documentation
    
    Be specific and complete in your response.
    Review your response and make corrections if required.

    IF YOU RECEIVE AN ERROR:
    - Report it as an error and do not write the report.


    RESPONSE TEMPLATE:
    Company Overview:
    - YOUR_RESPONSE
    Internet presence:
    - YOUR_RESPONSE
    Website and software:
    - YOUR_RESPONSE
    Summary:
    - YOUR_RESPONSE
    """
    combine_prompt_template = PromptTemplate(
        template=combine_prompt,
        input_variables=[
            "prospect",
            "text",
        ],
    )

    llm = ChatOpenAI(api_key=openai_api_key)

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=True,
    )

    output = chain(
        {
            "input_documents": docs,
            "prospect": company_name,
        }
    )

    print(output["output_text"])

    # Save the output to a file
    with open("docs/output.txt", "w", encoding="utf-8") as f:
        f.write(output["output_text"] + "\n")

    return output["output_text"]

    # module for iteration
    # for i, company in df_companies.iterrows():
    #    print (f"{i + 1}. {company['Name']}")
    #    page_data = get_company_page(company['Link'])
    #    docs = text_splitter.split_documents(page_data)

    #    output = chain({"input_documents": docs, \
    #                "company":"RapidRoad", \
    #                "sales_rep" : "Greg", \
    #                "prospect" : company['Name'],
    #                "company_information" : company_information
    #               })

    #    print (output['output_text'])
    #    print ("\n\n")


if __name__ == "__main__":
    print(run_marketing_report("allservicemoving.com"))
