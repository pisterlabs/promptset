import os
from langchain.chains import LLMChain 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from .models import create_llm_data

load_dotenv()

class ColumnMapper:
    """
    Singleton class for suggesting column mappings using OpenAI's LLM.
    
    Attributes:
        llm (OpenAI): Instance of the OpenAI's language model for performing tasks.
    """

    _instance = None
    
    def __new__(cls):
        """
        Create a new instance or return the existing one for the ColumnMapper class.
        """
        if cls._instance is None:
            try:
                cls._instance = super(ColumnMapper, cls).__new__(cls)
                
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("API key not found in .env file!")
                
                cls._instance.llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0, max_tokens=1500,api_key=api_key)
            except Exception as e:
                print(f"Error initializing ColumnMapper: {e}")
                return None  
            
        return cls._instance

    @staticmethod
    def get_column_info(df):
        """
        Extracts information from DataFrame's columns.
        
        Args:
            df (DataFrame): Pandas DataFrame to extract column information from.
        
        Returns:
            list: A list of dictionaries containing column names, data types, and sample values.
        """

        try:
            df = df.loc[:, ~df.T.duplicated(keep='first')]
            column_info_list = []
            for col in df.columns:
                column_info = {
                    "column_name": col,
                    "data_type": str(df[col].dtype),
                    "sample_values": df[col].sample(5).tolist()
                }
                column_info_list.append(column_info)
            return column_info_list
        except Exception as e:
            print(f"Error getting column info: {e}")
            return []

    def suggest_column_mapping(self, info_input, info_template,template,suggestion=None):
        """
        Suggests column mappings based on given input using OpenAI's LLM.
        
        Args:
            info_input (dict): Information about the input data.
            info_template (dict): Information about the template data.
            template (str): The prompt template for the LLM.
            suggestion (str, optional): Optional suggestion to aid the LLM. Defaults to None.
        
        Returns:
            dict: Dictionary containing suggestions and code provided by the LLM.
        """
        try:
            
            prompt_inputs = ["info_input", "info_template"]
            chain_inputs = {"info_input": info_input, "info_template": info_template}
            
            if suggestion:
                prompt_inputs.append("suggestion")
                chain_inputs["suggestion"]=suggestion


            prompt = PromptTemplate(template=template, input_variables=prompt_inputs)


            saving_prompt=prompt.format(**chain_inputs)

                

            llm_chain = LLMChain(
                prompt=prompt,
                llm=self.llm
            )
            

            response = llm_chain.run(chain_inputs)
            create_llm_data(prompt=saving_prompt,response=response)
            suggestion, code = response.split("Code:")

            return {"suggestion":suggestion,"code":code}
        except Exception as e:
            print(f"Error suggesting column mapping: {e}")
            return {"suggestion": "", "code": ""}


