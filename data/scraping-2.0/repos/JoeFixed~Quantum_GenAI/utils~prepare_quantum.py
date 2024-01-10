

'''import all the necessary libraries'''
from langchain import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from config import Settings


def perform_mapping(text):
        """
        Generates a mapping cluster for the provided text using the Large Language Model (LLM).

        Parameters:
            llm (LLM): An instance of the Large Language Model (LLM) for generating responses.
            text (str): The text that needs to be summarized.

        Returns:
            str: A cluster map for each text
        """

        # Create the template
        settings = Settings()

        template_string = '''You are the master of clustering and categorization ; please
        categorize the following text if it is represents ('Science' or 'Politics' or 'Information Technology' or 
        'Economics' or 'Culture' or 'Entertainment' or 'Media' or 'sports'):

        Input Text:
        "{input_text}"

        Prompt Template:
        "Please categorize the given text according to one of the following four categories ('Science' , 'Politics' ,'Information Technology','Economics' , Culture')"

        Summary:
        '''
        llm = OpenAI(max_tokens=-1,  openai_api_key=settings.GPT_API_KEY)
        # LLM call
        prompt_template = ChatPromptTemplate.from_template(template_string)
        chain = LLMChain(llm=llm, prompt=prompt_template)
        response = chain.run({"input_text" : text})

        return response

    




 

