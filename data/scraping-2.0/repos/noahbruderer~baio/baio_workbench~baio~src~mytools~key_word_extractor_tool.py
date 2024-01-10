from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI
from src.llm import LLM

llm35 = LLM.get_instance('gpt-3.5-turbo')

class NaturalLanguageExtractors:
    """This class contains methods to extract certain information in a structured manner from natural language."""

    def __init__(self, natural_language_string):
        """
        Initialize the extractor with the natural_language_string input by the user.
        
        Parameters:
        natural_language_string (str): The string from which the information has to be extracted.
        """
        self.natural_language_string = natural_language_string

    def gene_name_extractor(self) -> list:
        """
        Extracts gene names and returns a list.
        """
        schema = {
            "properties": {
                "gene_names": {"type": "string"},
            },
            "required": ["gene_names"],
        }

        # Input 
        # Run chain
        chain = create_extraction_chain(schema, llm35)
        result = chain.run(self.natural_language_string)

        #unpacking the list and splitting 
        gene_list = [gene.strip() for gene in [result[0]['gene_names']][0].split(',')]
        return gene_list


##Testing:
# question = 'i want the go annotations from YAP1, TAZ but also form HELLI'

# extractor = NaturalLanguageExtractors(question)
# gene_list = extractor.gene_name_extractor()
# print(gene_list)
