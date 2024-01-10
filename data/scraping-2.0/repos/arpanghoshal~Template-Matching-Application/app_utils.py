import pandas as pd
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

class AppUtils:
    def __init__(self):
        self.user_table_str_col = None
        self.user_table_str_frst = None
        self.template_table_str_col = None
        self.template_table_str_frst = None

    @staticmethod
    def load_csv(file):
        try:
            return pd.read_csv(file, nrows=5), None
        except Exception as e:
            return None, str(e)

    @staticmethod
    def run_llm_chain(llm, prompt, variables):
        try:
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run(variables), None
        except Exception as e:
            return None, str(e)

    def generate_transformation_instructions(self, user_file, template_file, openai_api_key):
        user_table, error = self.load_csv(user_file)
        template_table, error = self.load_csv(template_file)

        if error is not None:
            return None, error

        self.user_table_str_col = str(list(user_table.columns))
        self.user_table_str_frst = str(list(user_table.iloc[0]))
        self.template_table_str_col = str(list(template_table.columns))
        self.template_table_str_frst = str(list(template_table.iloc[0]))

        template_table_lst_col = list(template_table.columns)

        result = {}
        for item in template_table_lst_col:
            result[f"old_{item}_format"] = "format of the item in the user data"
            result[f"new_{item}_format"] = "format of the item in the template data"
            result[f"old_{item}_datatype"] = "data type of the item in the user data"
            result[f"new_{item}_datatype"] = "data type of the item in the template data"
            result[f"old_{item}_example_data"] = "first/example data of the item in the user data"
            result[f"new_{item}_example_data"] = "first/example data of the item in the template data"

        result = str(result)

        result = str(result)
        output_example = """
        {
            "column_renames": {},
            "columns_to_remove": [list of columns that to be removed],
            "columns_to_keep": [list of columns that to be kept],
            "data_transformations":  %s
        }
        """ % result
        describe_output_example = """
        Here column_renames is a mapping of user data columns to template data columns, for example:
        {
            "user_data_column_1": "template_data_column_1",
            "user_data_column_2": "template_data_column_2",
        }

        Here in data_transformations, while computing observe these very carefully:
        {
            "old_item_format": format of the item in the user data
            "new_item_format": format of the item in the template data,
            "old_item_datatype": data type of the item in the user data
            "new_item_datatype": data type of the item in the template data
            "old_item_example_data": example data of the item in the user data
            "new_item_example_data": example data of the item in the template data
        }

        item are the column name in the template data
        """
        prompt = PromptTemplate(
            input_variables=["user_table_str_col", "user_table_str_frst", "template_table_str_col", "template_table_str_frst", "output_example", "describe_output_example"],
            template="""
            Given the following information:
            - User_data columns: {user_table_str_col}
            - User_data example row: {user_table_str_frst}
            - Template_data columns: {template_table_str_col}
            - Template_data example row: {template_table_str_frst}

            Generate a JSON object detailing:
            1. Mapping of user data columns to template data columns (column_renames)
            2. Data transformations of format and data type, it also shows the example data of column (data_transformations) [CAUTION: data_transformations should fill all the values. If you don't want to change the format or datatype or firstdata, please fill the same value as the old one.
]
            3. Columns to remove or keep (columns_to_remove, columns_to_keep)

            The output should follow JSON format, this is one of the example: {output_example}
            
            {describe_output_example}
            """
        )

        llm = ChatOpenAI(model='gpt-3.5-turbo',openai_api_key=openai_api_key,temperature=0.5)
        return self.run_llm_chain(llm, prompt, {
            'user_table_str_col': self.user_table_str_col,
            'user_table_str_frst': self.user_table_str_frst,
            'template_table_str_col':self.template_table_str_col,
            'template_table_str_frst':self.template_table_str_frst,
            'output_example':output_example,
            'describe_output_example':describe_output_example
        })


    def generate_correction_instructions(self, json_output, not_correct, openai_api_key):
        llm = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=openai_api_key, temperature=0)
        prompt = PromptTemplate(
            input_variables=["json_output", "not_correct"],
            template="""
            - Update the following JSON output:
                {json_output}
            - Based on these corrections:
                {not_correct}
            """
        )
        return self.run_llm_chain(llm, prompt, {'json_output': json_output, 'not_correct': not_correct})

    def generate_transformation_code(self, json_output, openai_api_key):
        llm = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=openai_api_key, temperature=0)
        prompt = PromptTemplate(
            input_variables=["user_table_str_col", "user_table_str_frst", "template_table_str_col", "template_table_str_frst", "json_output"],
            template="""
            - The user data columns are: {user_table_str_col}
            - The user data example row is: {user_table_str_frst}
            - The template data columns are: {template_table_str_col}
            - The template data example row is: {template_table_str_frst}
            - Given these instructions in JSON format:
            {json_output}
            - Write a Python script transforming 'user_data.csv' to 'transformed_data.csv' using pandas to:
            1. Rename the columns according to 'column_renames'
            2. Drop the columns listed in 'columns_to_remove'
            3. Keep the columns listed in 'columns_to_keep'
            4. Apply the transformations specified in 'data_transformations'
            - The output should be only the python script
            """
        )
        return self.run_llm_chain(llm, prompt, {
            'user_table_str_col': self.user_table_str_col,
            'user_table_str_frst': self.user_table_str_frst,
            'template_table_str_col': self.template_table_str_col,
            'template_table_str_frst': self.template_table_str_frst,
            'json_output': json_output
        })
