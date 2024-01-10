from dotenv import load_dotenv
import os
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import pandas as pd

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

def get_company_page(company_path):
    url = input("Enter the URL of the company you're interested in: ")

    print (url)

    loader = UnstructuredURLLoader(urls=[url])
    return loader.load()


# Get the data of the company you're interested in
data = get_company_page("")

print (f"You have {len(data)} document(s)")

print (f"Preview of your data: /n/n{data[0].page_content[900:1150]}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 0
)

docs = text_splitter.split_documents(data)

print (f"You now have {len(docs)} documents")

map_prompt = """Below is a section of a website about {prospect}

Write a concise summary about {prospect}. If the information is not about {prospect}, exclude it from your summary.

{text}

CONCISE SUMMARY:"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "prospect"])

combine_prompt = """
Your goal is to write a personalized outbound email from {sales_rep}, a sales rep at {company} to {prospect}.

A good email is personalized and combines information about the two companies on how they can help each other.
Be sure to use value selling: A sales methodology that focuses on how your product or service will provide value to the customer instead of focusing on price or solution.

INFORMATION ABOUT {company}:
{company_information}

INFORMATION ABOUT {prospect}:
{text}

INCLUDE THE FOLLOWING PIECES IN YOUR RESPONSE:
- Start the email with the sentence: "We love that {prospect} helps teams..." then insert what they help teams do.
- The sentence: "We can help you do XYZ by ABC" Replace XYZ with what {prospect} does and ABC with what {company} does
- A 1-3 sentence description about {company}
- End your email with a call-to-action such as asking them to set up time to talk more

YOUR RESPONSE:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["sales_rep", "company", "prospect", \
                                                                         "text", "company_information"])

company_information = """"
- We are a SAS development company that has built a scalable cloud based software platform for management of business to residential services.
- Initially designed with movers in mind we developed the platform to be adjusted to fit any kind of business that provides services directly to residential homes.
- Scalable system that is competitively priced at all levels, from small to multinational independently owned businesses to corporations.
- Manage bookings and scheduling, user and location information, products and services and create custom rules to automatically estimate scheduled events.
- Human assisted onboarding that is here to help every step of the way.
- Active development with continuous improvements taken directly from the feedback of our customers.
"""

llm = OpenAI(temperature=.7, openai_api_key=openai_api_key)

chain = load_summarize_chain(llm,
                             chain_type="map_reduce",
                             map_prompt=map_prompt_template,
                             combine_prompt=combine_prompt_template,
                             verbose=True
                            )

output = chain({"input_documents": docs,
            "company": "Karve IT",
            "company_information" : company_information,
            "sales_rep" : "Richard Porteous",
            "prospect" : input("Prospect name: ")
           })

print(output['output_text'])

# Save the output to a file
with open ("../docs/output.txt", "a") as f:
    f.write(output['output_text'])


#module for iteration
#for i, company in df_companies.iterrows():
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