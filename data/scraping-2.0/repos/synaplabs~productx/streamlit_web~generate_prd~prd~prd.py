from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback, OpenAICallbackHandler
from langchain.document_loaders import PyPDFLoader

from generate_prd.vectordb.chromadb_client import ChromaDBClient
from generate_prd.web_search.serpapi_client import SerpApiClient
from generate_prd.llm.llm import LLM
import generate_prd.prompts.prompts as prompts

import os
import time
import json
import re
import wandb
from wandb.integration.langchain import WandbTracer


class PRD:
    """
    Product Requirements Document (PRD) generator.
    """

    def __init__(self, product_name: str, product_description: str, serpapi_api_key: str, input_prd_template_file_path: str = None) -> None:
        """
        Initialize PRD generator.

        Args:
            product_name (str): Product name.
            product_description (str): Product description.
            serpapi_api_key (str): SerpAPI API key.
            input_prd_template_file_path (str): Input PRD template file path - created using `tempfile.NamedTemporaryFile`

        Returns:
            None
        """
        self.product_name = product_name
        self.product_description = product_description
        self.serpapi_api_key = serpapi_api_key
        self.input_prd_template_file_path = input_prd_template_file_path
        self.document = ""
        self.cost = {
            "prd": {
                "cost": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            },
            "db": {
                "cost": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
        }

        self.LLM = LLM(chat_model="openai-gpt-4") 
        self.chain = self.LLM.chain
        # self.llm = self.LLM.llm
        
        self.chroma = ChromaDBClient(chromadb=Chroma(
            embedding_function=OpenAIEmbeddings()))
        self.serpapi = SerpApiClient(api_key=self.serpapi_api_key)

        wandb.init(
            project=f"{self.product_name}_prd_{self.chain.llm.model_name}",
            config={
                "model": self.chain.llm.model_name,
                "temperature": self.chain.llm.temperature,
            },
            entity="arihantsheth",
            name=f"{self.product_name}_prd_{self.chain.llm.model_name}",
        )

    def add_cost(self, callback: OpenAICallbackHandler, src: str):
        """
        Add cost to cost dictionary.

        Args:
            callback (OpenAICallbackHandler): OpenAICallbackHandler instance.
            src (str): Source of cost. Must be one of "prd" or "db".
        """
        if src not in ["prd", "db"]:
            raise ValueError("Invalid src value. Must be one of 'prd' or 'db'")
        
        self.cost[src]["cost"] += callback.total_cost
        self.cost[src]["prompt_tokens"] += callback.prompt_tokens
        self.cost[src]["completion_tokens"] += callback.completion_tokens

    def get_prompts_from_pdf(self):
        """
        Get prompts from PDF.

        Args:
            None

        Returns:
            None
        """
        loader = PyPDFLoader(self.input_prd_template_file_path)
        pages = loader.load_and_split()

        self.chroma.chromadb.add_documents(documents=pages)
        self.chroma.update_qa_chain(k=2)

        with get_openai_callback() as callback_get_prompts_from_pdf:
            pdf_prompts, source_docs = self.chroma.ask_with_source("""\
Given this document containing PRD requirements in paragraphs, generate a PRD template following the below given structure:

# PRD Template

## Section (give a name to the section)
Description of section. What is the purpose of this section? What is the expected output of this section? 

### Subsection Name (give a name to the subsection) - only create if needed

Table (if needed):
|          	| Column - 1 	| Column - 2 	| Add more 	|
|----------	|------------	|------------	|----------	|
| Item - 1 	|            	|            	|          	|
| Item - 2 	|            	|            	|          	|
| Add more 	|            	|            	|          	|
""")

        self.add_cost(src="db", callback=callback_get_prompts_from_pdf)

        pdf_prompt_sections = re.split(r'\n##\s+', pdf_prompts)[1:]
        pdf_prompt_list = ["## " + section.strip() for section in pdf_prompt_sections]

        our_local_prompts = "\n\n".join(prompts.LOCAL_PROMPTS_LIST)
        our_web_prompts = "\n\n".join(prompts.WEB_PROMPTS_LIST)

        with get_openai_callback() as callback_get_prompts_from_pdf:
            final_prompts = self.LLM.llm.predict(text=f"""\
There are 2 PRD generation prompt sets. Ours and PDF's.

Our prompts are divided into 2 categories: Local and Web.

Local prompts are the ones which do not require any external research.
Web prompts are the ones which require external data to be fed into the model.

PDF prompts are the ones which are extracted from the PDF.
Generate a super set of all of these prompts with only the important and unique prompts. 
Exclude prompts that are repeated and not needed.
For example, if both sets have similar prompts:
'Product description' and 'product summary', then only keep one of them.
Format the prompts in a markdown structure.
In addition, limit the total number of prompts to 11.
The first 6 prompts should be local prompts. They can be from either set.
The web prompts should be from 7 to 11. They have to be from the web prompts set.
The result should only have the list of prompts. No other text. No web/local prompt separation.

Incase of Market Research, Competitive Analysis, and Success Metrics from 'Our web prompts', \
use those prompts as it is. Do not change them. Only add to them if you think it is necessary.

Use the below given structure for the PRD template:               
# PRD Template

## Section (give a name to the section)
Description of section. What is the purpose of this section? What is the expected output of this section? 

### Subsection Name (give a name to the subsection) - only create if needed

Table (if needed):
|          	| Column - 1 	| Column - 2 	| Add more 	|
|----------	|------------	|------------	|----------	|
| Item - 1 	|            	|            	|          	|
| Item - 2 	|            	|            	|          	|
| Add more 	|            	|            	|          	|

Our local prompts:
{our_local_prompts}

Our web prompts:
{our_web_prompts}

PDF prompts:
{pdf_prompts}
""")
            
        self.add_cost(src="prd", callback=callback_get_prompts_from_pdf)

        final_prompts_sections = re.split(r'\n##\s+', final_prompts)[1:]
        final_prompts_list = ["## " + section.strip() for section in final_prompts_sections]
        self.final_local_prompts_list = final_prompts_list[:6]
        self.final_web_prompts_list = final_prompts_list[6:]

        with open("final_prompts.txt", "w") as f:
            f.write(final_prompts)
    
    def local_prompts(self, initial_prompt: str = prompts.INITIAL_PROMPT, local_prompts_list: list = prompts.LOCAL_PROMPTS_LIST):
        """
        Pass local (prompts without live data) prompts to the model.

        Args:
            initial_prompt (str): Initial prompt.
            local_prompts_list (list): List of local prompts.

        Returns:
            None
        """
        with get_openai_callback() as callback_local_prompts:
            _ = self.chain.predict(
                input=initial_prompt.format(
                    product_name=self.product_name, product_description=self.product_description),
                callbacks=[WandbTracer()],
            )

            for prompt in local_prompts_list:
                self.document += self.chain.predict(
                    input=prompt,
                    callbacks=[WandbTracer()],
                ) + "\n\n"

                print(
                    f"Local prompt {local_prompts_list.index(prompt) + 1} completed out of {len(local_prompts_list)}.")

        self.add_cost(src="prd", callback=callback_local_prompts)

    def get_comp_info(self, comp_search_query_prompt: str = prompts.COMP_SEARCH_QUERY_PROMPT, comp_queries: str = prompts.COMPETITOR_QUERIES):
        """
        1. Get competitor search query.
        2. Search for competitors.
        3. Get competitors list by retrieving names from ChromaDB.
        4. Get competitor info by searching
        5. Retrieve competitor info from ChromaDB.

        Args:
            comp_search_query_prompt (str): Competitor search query prompt.
            comp_queries (str): Competitor info queries.

        Returns:
            None
        """
        # Get competitor search query
        with get_openai_callback() as callback_get_comp_search_query:
            self.comp_search_query = self.chain.predict(
                input=comp_search_query_prompt,
                callbacks=[WandbTracer()],
            )
            print(f"Competitor search query: {self.comp_search_query}")

        self.add_cost(src="prd", callback=callback_get_comp_search_query)

        # Get competitors list
        comp_list_content = self.serpapi.webpages_from_serpapi(
            query=self.comp_search_query,
            num_results=3
        )
        self.chroma.store_web_content(content=comp_list_content)
        self.chroma.update_qa_chain(k=2)

        with get_openai_callback() as callback_get_competitors_list:
            self.competitors, self.competitors_source = self.chroma.ask_with_source(
                query=f"{self.comp_search_query}. Only return the names in a comma separated list (maximum 5)."
            )
            self.competitors = self.competitors.replace(" ", "").split(",")
            print(f"Competitors: {self.competitors}")
            
        self.add_cost(src="db", callback=callback_get_competitors_list)
        print(f"Competitors: {self.competitors}")

        # Get competitor info
        for competitor in self.competitors:
            print(f"Searching info for {competitor}")

            for query in comp_queries:
                self.serpapi.webpages_from_serpapi(
                    query=query.format(competitor=competitor),
                    num_results=3
                )
                self.chroma.store_web_content(content=comp_list_content)
                self.chroma.update_qa_chain(k=2)

        self.comp_analysis_results = {competitor: {}
                                      for competitor in self.competitors}
        query_names = ["User Base", "Revenue", "New Features"]

        for competitor in self.competitors:
            print(f"Retrieving info for {competitor}")

            with get_openai_callback() as callback_query_competitors_db:
                for query, dict_key in zip(comp_queries, query_names):
                    try:
                        answer, source_documents = self.chroma.ask_with_source(
                            query=query.format(competitor=competitor))
                    except Exception as e:
                        print(f"Error: {e}")
                        time.sleep(60)
                        continue

                    try:
                        self.comp_analysis_results[competitor][dict_key] = answer + \
                            "\n Web Source: " + \
                            source_documents[0].metadata['source']
                    except Exception as e:
                        print(f"Error: {e}")
                        self.comp_analysis_results[competitor][dict_key] = answer

            self.add_cost(src="db", callback=callback_query_competitors_db)
            self.comp_analysis_results_str = json.dumps(
                self.comp_analysis_results, indent=2)

    def get_metrics_info(self, metrics_search_query_prompt: str = prompts.METRICS_SEARCH_QUERY_PROMPT):
        """
        1. Get metrics search query.
        2. Search for metrics.
        3. Store metrics in ChromaDB.
        4. Retrieve metrics info from ChromaDB.

        Args:
            metrics_search_query_prompt (str): Metrics search query prompt.

        Returns:
            None
        """
        # Get metrics search query
        with get_openai_callback() as callback_get_metrics_search_query:
            self.metrics_search_query = self.chain.predict(
                input=metrics_search_query_prompt,
                callbacks=[WandbTracer()],
            )
            print(f"Metrics search query: {self.metrics_search_query}")

        self.add_cost(src="prd", callback=callback_get_metrics_search_query)

        # Search for metrics
        metrics_content = self.serpapi.webpages_from_serpapi(
            query=self.metrics_search_query,
            num_results=3
        )

        # Store metrics in ChromaDB
        self.chroma.store_web_content(content=metrics_content)
        self.chroma.update_qa_chain(k=2)

        # Retrieve metrics info
        with get_openai_callback() as callback_query_metrics_db:
            self.metrics_info, self.metrics_source = self.chroma.ask_with_source(
                query=f"{self.metrics_search_query}. Only return the names in a comma separated list (maximum 5 names)."
            )

        try:
            self.metrics_info += "\n Web Source: " + \
                self.metrics_source[0].metadata['source']
        except Exception as e:
            print(f"Error: {e}")

        self.add_cost(src="db", callback=callback_query_metrics_db)

    def web_prompts(self, web_prompts_list: list = prompts.WEB_PROMPTS_LIST):
        """
        Pass web (with realtime information) prompts to the model.

        Args:
            web_prompts_list (list): List of web prompts.

        Returns:
            None
        """
        with get_openai_callback() as callback_web_prompts:
            for prompt in web_prompts_list:
                try:
                    self.document += self.chain.predict(
                        input=prompt.format(
                            comp_analysis_results_str=self.comp_analysis_results_str, metrics_info=self.metrics_info),
                        callbacks=[WandbTracer()],
                    ) + "\n\n"
                except Exception as e:
                    print(f"Error: {e}")
                    print(
                        f"Web prompt {web_prompts_list.index(prompt) + 1} skipped out of {len(web_prompts_list)}.")
                    continue

                print(
                    f"Web prompt {web_prompts_list.index(prompt) + 1} completed out of {len(web_prompts_list)}.")

        self.add_cost(src="prd", callback=callback_web_prompts)
        wandb.finish()

    def save_prd(self):
        """
        Save PRD to file.

        Args:
            None

        Returns:
            None
        """
        with open(f"{self.product_name} PRD {self.chain.llm.model_name} web v1.1.5 2.md", "w") as f:
            f.write(self.document)
