import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


class Transformer:
    """
    Transformer class generates transformation code using OpenAI 
    """
    def __init__(self):
        self.template_columns = None 
        self.template_first_row = None
        self.source_columns = None
        self.source_first_row = None
    
    @staticmethod
    def read_csv(csv):
        try:
            csv = pd.read_csv(csv)
            return csv
        except Exception as e:
            return str(e)
    
    @staticmethod
    def run_llmchain(llm,prompt,args):
        try:
            chain = LLMChain(llm=llm,prompt=prompt)
            return chain.run(args)
        except Exception as e:
            return str(e)
    
    def generate_transformations(self,template_upload,csv_upload,openai_api_key):
        """
        Generates transformations as per template
        """
        template = self.read_csv(template_upload)
        source = self.read_csv(csv_upload)
        
        self.template_columns = template.columns.tolist()
        self.template_first_row = template.iloc[0].tolist()
        self.source_columns = source.columns.tolist()
        self.source_first_row = source.iloc[0].tolist()
        
        output = {}

        for col in self.template_columns:
            output[f"old_source_{col}_format"] = "format of the column in the source user data"
            output[f"new_template_{col}_format"] = "format of the column in the template data"
            output[f"old_source_{col}_datatype"] = "data type of the column in the source user data"
            output[f"new_template_{col}_datatype"] = "data type of the column in the template data"
            output[f"old_source_{col}_example_data"] = "example data of the column in the source user data"
            output[f"new_template_{col}_example_data"] = "example data of the column in the template data"
        
        example = """
        {
            "renamed_columns_as_per_template": {},
            "kept_columns_as_per_template": [list of columns to be kept]
            "removed_columns_as_per_template": [list of columns to be removed],
            "transformations":  %s
        }
        """ % output

        detailed_example = """
        Here renamed_columns_as_per_template is a mapping of source data columns to template data columns, for example:
        {
            "source_data_column_1": "template_data_column_1",
            "source_data_column_2": "template_data_column_2",
        }

        Here in transformations, while computing observe these very carefully:
        {
            "old_source_col_format": format of the column in the source user data
            "new_template_col_format": format of the column in the template data
            "old_source_col_datatype": data type of the column in the source user data
            "new_template_col_datatype": data type of the column in the template data
            "old_source_col_example_data": example data of the column in the source user data
            "new_template_col_example_data": example data of the column in the template data
        }

        col is the column name in the template data
        """

        prompt = PromptTemplate(
            input_variables=["source_columns", "source_first_row", "template_columns", "template_first_row", "example", "detailed_example"],
            template="""
            Given the following information:

            - Source_data columns: {source_columns}
            - Source_data first row example: {source_first_row}
            - Template_data columns: {template_columns}
            - Template_data first row example: {template_first_row}

            Generate a JSON object detailing:

            1. Mapping of source data columns to template data columns (renamed_columns_as_per_template)
            
            2. Data transformations of format and data type as per {template_first_row}, it also shows the example data of column (transformations) [CAUTION: transformations should fill all the values. If you don't want to change the format or datatype or firstdata, please fill the same value as the old one.]
            3. Columns to remove or Columns to keep (removed_columns_as_per_template, kept_columns_as_per_template)

            The output should follow JSON format, this is one of the example: {example}
            
            {detailed_example}
            
            """
        )

        llm = ChatOpenAI(model='gpt-3.5-turbo',openai_api_key=openai_api_key,temperature=0.2)
        return self.run_llmchain(llm, prompt, {
            'source_columns': self.source_columns,
            'source_first_row': self.source_first_row,
            'template_columns':self.template_columns,
            'template_first_row':self.template_first_row,
            'example':example,
            'detailed_example':detailed_example
        })
        
    
    def feedback(self,transformation,feedback,openai_api_key):

        """ Generates updated transformation as per feedback """

        llm = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=openai_api_key, temperature=0)
        prompt = PromptTemplate(
            input_variables=["transformation", "feedback"],
            template="""
            - Update the following JSON output:
                {transformation}
            - Based on these corrections carefully:
                {feedback}
            """
        )
        return self.run_llmchain(llm, prompt, {'transformation': transformation, 'feedback': feedback})
    
    def transformations_to_code(self,transformation,openai_api_key):
        ''' Generates python code as per transformation'''

        llm = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=openai_api_key, temperature=0)
        prompt = PromptTemplate(
            input_variables=["source_columns", "source_first_row", "template_columns", "template_first_row", "transformation"],
            template="""
            - The source data columns are: {source_columns}
            - The source data example row is: {source_first_row}
            - The template data columns are: {template_columns}
            - The template data example row is: {template_first_row}
            - Given these instructions in JSON format:
            
            {transformation}
            
            - Write a Python script transforming 'source_data.csv' to 'transformed_data_as_per_template.csv' using pandas to:
            
            1. Rename the columns according to 'renamed_columns_as_per_template'
            2. Drop the columns listed in 'removed_columns_as_per_template'
            3. Keep the columns listed in 'removed_columns_as_per_template'
            4. Apply the transformations specified in 'transformations'
            
            - The output should only be python script compatible with python3.
            
            """
        )
        return self.run_llmchain(llm, prompt, {
            'source_columns': self.source_columns,
            'source_first_row': self.source_first_row,
            'template_columns':self.template_columns,
            'template_first_row':self.template_first_row,
            'transformation': transformation
        })


        
