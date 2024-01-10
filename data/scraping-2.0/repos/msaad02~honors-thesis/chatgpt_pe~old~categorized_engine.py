"""
This script provides an end-to-end question-answering pipeline that utilizes multiple techniques
for categorizing and answering questions. It integrates OpenAI's GPT models and HuggingFace's 
BgeEmbeddings to retrieve relevant information.

Global Parameters:
    - QA_OR_PICK_MAX_TOKENS: The max number of tokens to generate when querying GPT for answers.
    - QA_OR_PICK_TEMPERATURE: The temperature for generating answers or picking categories.
    - CATEGORIZATION_MAX_TOKENS: Max tokens for generating category choices.
    - CATEGORIZATION_TEMPERATURE: Temperature for generating category choices.
    - categorized_chroma_path: Directory for CHROMADB categorized data.

Functions:
    - query_gpt: Queries a GPT model and returns its response.
    - get_relevant_docs_for_dir: Retrieves relevant documents from ChromaDB for a given question.
    - answer_question_with_categorization: Orchestrates the entire process of question answering
      using a categorized search engine.

Usage:
    This script is intended to be used as a library, imported into other Python applications.
    Initialize the main class, and use its `__call__()` method to get an answer to a question.

Example:
    >>> from chatgpt_pe.categorized_engine import QuestionAnswering
    >>> qa_bot = QuestionAnswering(param1=value1, param2=value2, ...)
    >>> answer = qa_bot("How can I apply to SUNY Brockport?")
    >>> print(answer)
"""

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
import openai
import json
from colorama import init as colorama_init
from colorama import Fore, Style
from torch.cuda import is_available
colorama_init()


class QuestionAnswering:
    def __init__(
            self, 
            base_path: str = "/home/msaad/workspace/honors-thesis/",
            qa_or_pick_max_tokens=256, 
            qa_or_pick_temperature=0.8, 
            categorization_max_tokens=10, 
            categorization_temperature=0, 
            categorized_chroma_path: str = None,
            categorized_data_path: str = None,
            verbose=False
        ):
        self.QA_OR_PICK_MAX_TOKENS = qa_or_pick_max_tokens
        self.QA_OR_PICK_TEMPERATURE = qa_or_pick_temperature
        self.CATEGORIZATION_MAX_TOKENS = categorization_max_tokens
        self.CATEGORIZATION_TEMPERATURE = categorization_temperature
        self.verbose = verbose

        if categorized_chroma_path is None:
            categorized_chroma_path = f"{base_path}/data_collection/data/categorized_datastore/chroma_data/"

        if categorized_data_path is None:
            categorized_data_path = f"{base_path}/data_collection/data/categorized_data.json"

        self.categorized_chroma_path = categorized_chroma_path
        self.categorized_data_path = categorized_data_path

        self.data = self._load_json_data()
        self.semantic_model = self._load_semantic_model()
        
    def _load_json_data(self):
        """
        Load JSON data from the specified file path.

        Returns:
            dict: Parsed JSON data.
        """
        with open(self.categorized_data_path, "r") as f:
            return json.load(f)

    def _load_semantic_model(self):
        """
        Initialize a HuggingFace BgeEmbeddings model. Can pick any model from the HuggingFace.

        Returns:
            HuggingFaceBgeEmbeddings: Initialized model.

        Notes:
            - Uses CUDA if available, otherwise defaults to CPU.
            - Embeddings are normalized. (Computes Cosine Similarity)
        """

        return HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={'device': 'cuda'} if is_available() else {'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _query_gpt(
            self, 
            system: str, 
            prompt: str,
            model_name: str = "gpt-4",
            temperature: float = 0.8,
            max_tokens: int = 256,
            price: float = 0
        ):
        """
        Queries an OpenAI GPT model and returns the generated response and associated cost.

        Parameters:
            - system (str): System-level instruction for the GPT model.
            - prompt (str): User prompt to generate a response for.
            - model_name (str, optional): Specifies which GPT model to use. Default is "gpt-4".
            - temperature (float, optional): Controls randomness in output. Default is 0.8.
            - max_tokens (int, optional): Maximum tokens in the output. Default is 256.
            - price (float, optional): Accumulated price of querying the model. Default is 0.

        Returns:
            tuple: Generated response from the model and the updated cost of the operation.

        Notes:
            - GPT-4 and GPT-3.5-turbo have different pricing which is handled within this function.
            - The `price` parameter allows for accumulating costs over multiple queries.
        """

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Calculate pricing as of 8/31/2023
        if model_name == "gpt-4":
            price = price + response['usage']['prompt_tokens'] * 0.03/1000 + response['usage']['completion_tokens'] * 0.06/1000
        elif model_name == "gpt-3.5-turbo":
            price = price + response['usage']['prompt_tokens'] * 0.0015/1000 + response['usage']['completion_tokens'] * 0.002/1000

        try:
            return eval(response.to_dict()['choices'][0]['message']['content']), price
        except:
            try:
                return response.to_dict()['choices'][0]['message']['content'], price
            except:
                return "Error: Couldn't parse response.", price


    def _get_relevant_docs_for_dir(self, path, question, search_args):
        """
        Return relevant documents for a given question and category (directory).

        Incase this is not clear, every single categorized directory has their own independent ChromaDB.
        This function uses the matched ChromaDB (indicated by path), and returns the relevant documents for a given question.
        """

        retriever = Chroma(
            persist_directory = path, 
            embedding_function = self.semantic_model
        ).as_retriever(search_kwargs = search_args)

        # Retreive the relevant documents using retreiver
        return retriever.get_relevant_documents(question)
    
    def __call__(
            self, 
            question,
            search_args = {"score_threshold": 0.8, "k": 3},
            verbose: bool = False # Recommended to set True for command line use
        ):
        """
        This function handles the entire process of answering a question with the categorized search engine.
        
        Input a question, output an answer.
        """
        verbose = verbose if verbose is not False else self.verbose # Default no, but if verbose passed in on init or here use that.
        price = 0

        # --------------------------------------------------------------------
        # FIRST PROMPT/INITIAL CATEGORIZATION

        initial_categorization_system = "You are a helpful classification system. Categorize a question into its category based on the brief description provided."
        initial_categorization_prompt = f"""\
        The question is: 
        {question}

        The following categories available are:
        "none": if the question does not fit into any of the above categories, or are not related to SUNY Brockport
        "live": for policy related questions
        "academics": academic related information to majors, or programs.
        "support": current student and faculty support
        "life": information about student life
        "about": information about the university, such as Title IX, mission statement, diversity, or strategic plan, local area, president, etc.
        "admissions": for prospective students looking to apply
        "graduate": information about graduate programs
        "admissions-aid": information about admissions and financial aid
        "scholarships-aid": information about scholarships and financial aid
        "library": information about the library
        "research-foundation": information about the research at Brockport

        Respond ONLY with the name of the category. (i.e. live, academics, etc.). If a question does not fit into any of the above categories, or is otherwise inappropriate, respond with "none"."""

        first_category, price = self._query_gpt(
            system=initial_categorization_system, 
            prompt=initial_categorization_prompt,
            temperature=self.CATEGORIZATION_TEMPERATURE,
            max_tokens=self.CATEGORIZATION_MAX_TOKENS,
            price=price
        )

        if verbose:
            print(f"\n{Fore.GREEN}FIRST CATEGORY: {Style.RESET_ALL}{first_category}")

        # Check if first category is "none" (unrelated)
        if first_category == "none":
            if verbose:
                print(f"\n{Fore.RED}ERROR: QUESTION IS UNRELATED.")
                print(f"\n{Fore.CYAN}FINAL PRICE: {Style.RESET_ALL}${round(price, 6)}\n\n")

            return "Error: Question is unrelated."

        # Now lets get the list of subcategories for the first category
        subcategory_keys = self.data[first_category].keys() # NOTE: This includes URLs AND subcategories. Need to filter out URLs
        subcategories = [non_url for non_url in subcategory_keys if not non_url.startswith("http")] # Filters out URLs
        print_pretty_subcategories = "\n".join(subcategories)

        path_to_first_category = f"{self.categorized_chroma_path}{first_category}/"
        vector_search_results = self._get_relevant_docs_for_dir(question=question, path=path_to_first_category, search_args=search_args)
        print_pretty_vector_search_results = "\n".join([doc.page_content for doc in vector_search_results])

        if verbose:
            print(f"\n{Fore.GREEN}PATH TO FIRST CATEGORY: {Style.RESET_ALL}{path_to_first_category}")
            print(f"\n{Fore.BLUE}VECTOR SEARCH RESULTS: \n{Style.RESET_ALL}{print_pretty_vector_search_results}")

        # --------------------------------------------------------------------
        # SECOND PROMPT/SUBCATEGORIZATION

        # This is a checker for if there are not enough search results, and we need to force a new category (if one available)
        non_category_keys = [url for url in subcategory_keys if url.startswith("http")]

        if len(non_category_keys) == 0:

            # Check if subcategories available or not:
            if len(subcategories) == 0:
                if verbose:
                    print(f"\n{Fore.RED}ERROR: NOT ENOUGH SEARCH RESULTS. NO SUBCATEGORIES AVAILABLE.")
                    print(f"\n{Fore.CYAN}FINAL PRICE: {Style.RESET_ALL}${round(price, 6)}\n\n")

                return "Error: Not enough search results and no subcategories available."

            # Force pick subcategory if only one available and move to final step
            elif len(subcategories) == 1:
                if verbose:
                    print(f"\n{Fore.RED}WARNING: NOT ENOUGH SEARCH RESULTS. ONLY ONE SUBCATEGORY AVAILABLE. FORCE PICKING SUBCATEGORY.")

                # Manually pick subcategory, so no need to query GPT
                subcategory_or_answer = subcategories[0]
                query_gpt_response = False

            # Query with no data available, only categories listed to GPT
            elif len(subcategories) > 1:
                if verbose:
                    print(f"\n{Fore.RED}WARNING: NOT ENOUGH SEARCH RESULTS. FORCE PICK SUBCATEGORY.")

                subcategory_system = "You are a helpful assistant. Given a question and set of categories, you will choose the most relevant category for the question."
                subcategory_prompt = f"""\
                The question is:
                {question}

                The following subcategories available are:
                {print_pretty_subcategories}

                Respond ONLY with the name of the chosen category. (i.e. "live", "academics", etc.)."""

                # Should respond with subcategory. NOT an answer.
                subcategory_temperature = self.CATEGORIZATION_TEMPERATURE
                subcategory_max_tokens = self.CATEGORIZATION_MAX_TOKENS
                query_gpt_response = True

        # Has data available, so continue normally and check if subcategories are available or not.
        else:
            if len(subcategories) == 0:
                # Subcategories NOT available. Do not include that portion prompt.

                print(f"\n{Fore.RED}WARNING: NO SUBCATEGORIES AVAILABLE.")
                
                subcategory_system = "You are a helpful assistant. Given a context and a question, if possible, you will answer the question."
                subcategory_prompt = f"""\
                The question is:
                {question}

                If the answer is available in the following information, answer the question. If not, refuse to answer the question.

                The information is:
                {print_pretty_vector_search_results}"""

                subcategory_temperature = self.QA_OR_PICK_TEMPERATURE
                subcategory_max_tokens = self.QA_OR_PICK_MAX_TOKENS
                query_gpt_response = True

            else:
                # Subcategories ARE available. Include the full prompt.
                subcategory_system = "You are a helpful assistant. Given a context and a question, you will either answer the question, or choose the most relevant category for the question."
                subcategory_prompt = f"""\
                The question is:
                {question}

                If the answer is available in the following data, answer the question. If not, choose one of the following subcategories.
                The decision is yours whether to search more, or take your current information. If a subcategory seems promising, chose that!
                Most likely, a specific subcategory will be more relevant than the general category.

                The current information is:
                {print_pretty_vector_search_results}

                The following subcategories available are:
                {print_pretty_subcategories}

                If you choose a subcategory, respond ONLY with the name of that category. (i.e. "live", "academics", etc.)."""

                subcategory_temperature = self.QA_OR_PICK_TEMPERATURE
                subcategory_max_tokens = self.QA_OR_PICK_MAX_TOKENS
                query_gpt_response = True

        if query_gpt_response:
            subcategory_or_answer, price = self._query_gpt(
                system=subcategory_system, 
                prompt=subcategory_prompt,
                temperature=subcategory_temperature,
                max_tokens=subcategory_max_tokens,
                price=price
            )

        # Naive way of checking if GPT chose a category or gave an answer. Works well surprisingly.
        if ' ' in subcategory_or_answer:
            if verbose:
                print(f"\n{Fore.RED}BREAKOUT ANSWER: \n{Style.RESET_ALL}{subcategory_or_answer}")
                print(f"\n{Fore.CYAN}FINAL PRICE: {Style.RESET_ALL}${round(price, 6)}\n\n")
            return subcategory_or_answer
        

        ### Continues past this point only if GPT chose a category

        # Pull data from chosen category
        path_to_final_category = f"{path_to_first_category}{subcategory_or_answer}"
        vector_search_results = self._get_relevant_docs_for_dir(question=question, path=path_to_final_category, search_args=search_args)
        print_pretty_vector_search_results = "\n".join([doc.page_content for doc in vector_search_results])

        if verbose:
            print(f"\n{Fore.GREEN}SUBCATEGORY: {Style.RESET_ALL}{subcategory_or_answer}")
            print(f"\n{Fore.GREEN}PATH TO FINAL CATEGORY: {Style.RESET_ALL}{path_to_final_category}")
            print(f"\n{Fore.BLUE}VECTOR SEARCH RESULTS: \n{Style.RESET_ALL}{print_pretty_vector_search_results}")


        # --------------------------------------------------------------------
        # THIRD AND FINAL POSSIBLE PROMPT

        final_system = "You are a helpful assistant. Given a context and a question, you will either answer the question, or refuse to answer the question."
        final_prompt = f"""\
        The question is:
        {question}

        If the answer is available in the following data, answer the question. If not, refuse to answer the question.

        The current information is:
        {print_pretty_vector_search_results}"""


        #### BEGIN LOGIC AFTER THIRD PROMPT
        answer, price = self._query_gpt(
            system=final_system, 
            prompt=final_prompt,
            temperature=self.QA_OR_PICK_TEMPERATURE,
            max_tokens=self.QA_OR_PICK_MAX_TOKENS, # ARBITRARY. Need to up incase it wants to answer the question.
            price=price
        )

        if verbose:
            # print(f"\n{Fore.MAGENTA}FINAL ANSWER: \n{Style.RESET_ALL}{answer}")
            print(f"\n{Fore.CYAN}FINAL PRICE: {Style.RESET_ALL}${round(price, 6)}")

        return answer