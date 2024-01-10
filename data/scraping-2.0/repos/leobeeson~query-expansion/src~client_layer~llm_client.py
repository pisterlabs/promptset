import tiktoken


from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.chat_models import ChatOpenAI


from src.dal.category_metadata import CategoryMetadata
from src.dal.category_audit import CategoryAudit
from src.loggers.log_utils import setup_logger


logger = setup_logger(__name__)


class LLMCLient:


    def __init__(self) -> None:
        self.parser = PydanticOutputParser(pydantic_object=CategoryMetadata)
        self.parser_audit = PydanticOutputParser(pydantic_object=CategoryAudit)
        self.prompt_audit = self.generate_audit_prompt()
        self.prompt = self.generate_prompt()


    def calculate_number_of_tokens(self, string: str, encoding_name: str = "cl100k_base") -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens


    def generate_audit_prompt(self) -> None:
        parser_audit_instructions = self.parser_audit.get_format_instructions()

        task_instructions = """
        You are a market analyst working for a company that provides market intelligence for search engine optimization.
        Your task is to:
        1) First understand a conceptual model for a market category defined by the following market category taxonomy:
        CATEGORY_TAXONOMY:
        ```{category_taxonomy}```
        2) Second, review every search query in the following list below:
        SEARCH QUERIES:
        ```{search_queries}```
        3) Third, for every search query, assign a relevance score between 0 and 1, where 0 means that the search query is not relevant at all to the above market category, and 1 means that it is very relevant to the market category.
        Please structure the search query-relevance pairs in the following format below, without modifying the search queries nor leaving any search query out:
        RESPONSE FORMAT:
        ###{format_instructions}###
        """

        prompt = PromptTemplate(
            template=task_instructions,
            input_variables=["category_taxonomy", "search_queries"],
            partial_variables={"format_instructions": parser_audit_instructions},
            output_parser=self.parser
        )

        return prompt
    

    def audit_queries(self, category_taxonomy: str, search_queries: list[str]) -> str:
        # prompt = self.generate_audit_prompt()
        chat = ChatOpenAI(temperature=0)
        llm_chain = LLMChain(
            llm=chat,
            prompt=self.prompt_audit
        )
        try:
            response = llm_chain.predict(category_taxonomy=category_taxonomy, search_queries=search_queries, return_only_outputs=True)
        except Exception as e:
            raise
        return response


    def generate_prompt(self) -> None:
        parser_instructions = self.parser.get_format_instructions()

        task_instructions = """
        You are a market analyst working for a company that provides market intelligence for search engine optimization.
        Your task is to:
        1) First understand a conceptual model for the market category encompassed by the search queries below surrounded in triple backticks
        2) Second, generate a taxonomy that best clusters the category into its most relevant subcategories.
        3) Third, for each subcategory, generate a label, description, list of subtopics, and a list of only the most relevant noun phrases.
        Keep all labels and subtopics to ideally three words long, and a maximum of five words Long.
        Keep all noun phrases to ideally two words long, and a maximum of three words long, and a maximum of ten noun phrases per subcategory.
        SEARCH QUERIES:
        ```{search_queries}```
        Please structure the taxonomy and its components in the following format below:
        RESPONSE FORMAT:
        ###{format_instructions}###
        """

        prompt = PromptTemplate(
            template=task_instructions,
            input_variables=["search_queries"],
            partial_variables={"format_instructions": parser_instructions},
            output_parser=self.parser
        )

        return prompt

    
    def generate_taxonomy(self, search_queries: list[str]) -> str:
        prompt = self.generate_prompt()
        chat = ChatOpenAI(temperature=0)
        llm_chain = LLMChain(
            llm=chat,
            prompt=prompt
        )
        try:
            response = llm_chain.predict(search_queries=search_queries, return_only_outputs=True)
        except Exception as e:
            raise
        return response


    def parse_taxonomy(self, response: str) -> CategoryMetadata:
        parsed_response = self.parser.parse(response)
        return parsed_response


    def fix_parsing(self, unparsed_response: str) -> CategoryMetadata:
        fixer_parser = OutputFixingParser.from_llm(parser=self.parser, llm=ChatOpenAI(temperature=0))
        fixed_response = fixer_parser.parse(unparsed_response)
        return fixed_response


    def fix_parsing_audit(self, unparsed_response: str) -> CategoryMetadata:
        fixer_parser = OutputFixingParser.from_llm(parser=self.parser_audit, llm=ChatOpenAI(temperature=0))
        fixed_response = fixer_parser.parse(unparsed_response)
        return fixed_response


    def populate_prompt(self, search_queries_compact: str) -> str:
        populated_prompt = self.prompt.format_prompt(search_queries=search_queries_compact)
        return populated_prompt


if __name__ == "__main__":
    import os
    os.chdir("/home/fbe/drive/projects/query_expansion")
    llm_client = LLMCLient()
    llm_input = llm_client.prompt.format_prompt(search_queries="hello world")
    encoding = tiktoken.get_encoding("cl100k_base")
    llm_client.calculate_number_of_tokens(llm_input.to_string(), encoding_name="cl100k_base")