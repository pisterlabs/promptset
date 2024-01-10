from dotenv import load_dotenv
import json
import requests
from bs4 import BeautifulSoup
from typing import List
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import pandas as pd
import os

# Required to load hidden variables from the environment
# In this case it is the API key for OpenAI
load_dotenv()


class CohortQualifier:
    """
    The `CohortQualifier` class is a Python class that uses a language model to classify companies into
    cohorts based on their descriptions.
    """

    def __init__(self):
        ### SETUP FOR BRUTE FORCE CLASSIFICATION ###

        # Defined the standard prompt which tells the llm what to do
        self.prompt_scaffold = """
            You are a financial analyst. You need to assign a cohort according to the provided description of the company. From the list below choose which cohort best suits the company's description and return the result as a json file without any explanations or additional writing.

            cohorts list = [
                "AdTech",
                "Advertising Services",
                "Agencies / System Integrators",
                "Digital Marketing Agencies",
                "IT Services",
                "Management Consulting",
                "AI",
                "Conversational AI", 
                "Machine Vision",
                "Retail AI",
                "Business Intelligence",
                "AR / VR",
                "Automotive",
                "Customer Acquisition and Relationship Management (Automotive)",
                "Dealer Management Systems (Automotive)",
                "Dealership Management (Automotive)",
                "F&I (Automotive)",
                "F&I Technology (Automotive)",
                "Fixed Operations (Automotive)",
                "Inventory Management (Automotive)",
                "Inventory, Auctions & Reconditioning (Automotive)",
                "Listing Platforms (Automotive)",
                "Managed Marketplaces (Automotive)",
                "P2P Marketplaces (Automotive)",
                "Peer-to-Peer & Subscription (Automotive)",
                "Registration and Titling (Automotive)",
                "Rental & Subscription (Automotive)",
                "Business Process Outsourcing",
                "CPG",
                "Credit Union",
                "Crypto / Blockchain",
                "DNVB",
                "Health & Beauty",
                "Merchandising",
                "Alternative Lenders",
                "Authentication Software",
                "B2B Payments",
                "BNPL",
                "Consumer FinTech",
                "Cross-Border Payments",
                "Digital Wallets",
                "Financial Management Software",
                "Infrastructure (FinTech)", 
                "Neo-banks",
                "Payment Processor / POS",
                "Government Tech",
                "FoodTech",
                "Grocery Tech", 
                "Packaging",
                "Real Estate / Mall Operators",
                "IoT",
                "Inventory Management",
                "Logistics & Supply Chain",
                "Mobility",
                "Leisure & Travel",
                "Marketplaces",
                "Travel & Leisure",
                "CRM",
                "CXP Platforms",
                "Event Management",
                "Experiential Players",
                "Loyalty",
                "MarTech",
                "Personalization Software",
                "Sales Enablement Software",
                "Cybersecurity",
                "Diversified / Others",
                "Financial Software",
                "Robotics & Drones",
                "InsurTech",
                "Media",
                "Product Information Management Software",
                "Print on Demand",
                "Gadgets",
                "Games / Sports / Toys", 
                "Recommerce",
                "Rental",
                "Beverages",
                "Catering",
                "Restaurant, Hospitality and Local Delivery Technology",
                "Ghost/Commercial Kitchens",
                "Restaurants",
                "Apparel",
                "E-commerce",
                "Luxury & Accessories",
                "Retailers",
                "Home Goods",
                "Data Analytics",
                "Data Management",
                "Retail Data",
                "Enterprise Search",
                "Social Platform / Software",
                "In-Store Technologies",
                "Robotics & Drones",
                "Software",
                "Professional Services",
                "HRTech",
                "Staffing",
                "Tech-enabled Services",
                "Workforce Management",
                "Telecommunications",
                "Ticketing and Events",
                "Trade Show Providers",
                "Utilities",
                "Utility and Energy Tech",
                "Web 3.0",
                "Gaming",
                "Data Integration",
                "Data Services",
                "Financial Institution", 
                "Financial Investors",
                "Corporate Venture Capital"
            ]

            description = {description}
            
            Example response = {{res: AdTech}}

            If you are unable to identify which cohort the company belongs to respond with "Not sure". Example = {{res: Not sure}}
        """

        # Create a prompt template which will be used to generate the prompt
        # It is specialised code from langchain which makes life easier
        self.prompt_template = PromptTemplate(
            input_variables=["description"],
            template=self.prompt_scaffold,
        )

        # Define the variable that holds the large language model
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        ### SETUP FOR SEMANTIC CLASSIFICATION ###
        with open("cohorts.json") as f:
            cohorts = json.load(f)

        # I'm not going for the most efficient way here
        # Set up data I wanna embed into v-db

        # Initialize an empty list to store DataFrames
        dfs = []
        # Iterate over cohorts.items()
        for cohort, description in cohorts.items():
            # Create a DataFrame for the current cohort and description
            temp_df = pd.DataFrame({"cohort": [cohort], "description": [description]})
            # Append the DataFrame to the list
            dfs.append(temp_df)

        # Use pd.concat to concatenate the list of DataFrames
        df = pd.concat(dfs, ignore_index=True)
        df["concatenated"] = df["cohort"].astype(str) + ": " + df["description"]

        # Set up v-db
        cohorts = df["cohort"].values
        descriptions = df["concatenated"].values
        text_splitter = CharacterTextSplitter(chunk_size=170, chunk_overlap=0)
        documents: List[Document] = []
        for cohort, description in zip(cohorts, descriptions):
            texts = text_splitter.split_text(description)
            for text in texts:
                documents.append(
                    Document(page_content=text, metadata={"cohort": cohort})
                )

        embeddings = OpenAIEmbeddings()
        self.docsearch = Chroma.from_documents(documents, embeddings)

    def remove_duplicates_preserve_order(self, input_list: List[str]) -> List[str]:
        unique_items = []
        for item in input_list:
            if item not in unique_items:
                unique_items.append(item)
        return unique_items

    def brute_classify(self, description: str, source: str) -> None:
        with get_openai_callback() as cb:
            # Define langchain chain which will run the prompt
            chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

            # Result of the prompt after running it
            res = chain.run(description=description)

            return {
                "result": {"result": json.loads(res)["res"]},
                "source": source,
                "total_cost": cb.total_cost,
            }

    def scemantic_classify(self, description: str, source: str) -> None:
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            chain_type="stuff",
            retriever=self.docsearch.as_retriever(),
            return_source_documents=True,
        )

        with get_openai_callback() as cb:
            result = qa(
                {
                    "query": f"""
                You are a financial analyst. Which cohort best suits a company of the following description. Return only the cohort name, without any explantions.
                    
                description = {description}
                
                Example response = "Consumer Fintech"            
                """
                }
            )

            return {
                "result": result,
                "source": source,
                "total_cost": cb.total_cost,
            }

    def get_website_text(self, url: str) -> str:
        web_address = url

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }

        # Check if the provided url includes a protocol
        if not url.startswith("http://") and not url.startswith("https://"):
            # If not, assume it's https and prepend it
            web_address = "https://www." + url

        # See if we can get a response from the website
        # depending on protocol
        try:
            response = requests.get(
                web_address, headers=headers, allow_redirects=True, timeout=10
            )
        except Exception as e:
            try:
                web_address = (
                    "http://www." + url
                    if not url.startswith("http://") and not url.startswith("https://")
                    else url
                )
                response = requests.get(
                    web_address, headers=headers, allow_redirects=True, timeout=10
                )
            except Exception as e:
                return f"Error - {e}"

        # If successful, parse the response
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            # Boil down the empty spaces and tabs
            res = soup.get_text()
            res = res.replace("\t", "")
            res = res.encode("unicode_escape").decode()
        else:
            return f"Error - {response.status_code}"

        # Remove total duplicates and empty lines
        result = self.remove_duplicates_preserve_order(res.split("\\n"))
        result = [s for s in result if s.strip() != ""]

        # Remove duplicates that are more than 70% similar
        for i in result:
            length = round(len(i) * 0.7)

            for j in result:
                if i[:length] == j[:length] and i != j:
                    result.remove(j)

        # Get rid of lines which are single words
        result = [s for s in result if " " in s]

        # Return a single string
        return " ".join(result)
