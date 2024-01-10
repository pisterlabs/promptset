import json
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts import PromptTemplate

class Summarizer():
    def __init__(self, model="gpt-3.5-turbo") -> None:
        self.model = ChatOpenAI(model_name=model)

    def find_new_metrics_prompt(self, category):

        prompt_files = {
            "Market Dataset": 'autofinviz/components/summarizer/prompts/market_summarizer.tmpl',
            "Economic Dataset": 'autofinviz/components/summarizer/prompts/economic_summarizer.tmpl',
            "Corporate Financial Dataset": 'autofinviz/components/summarizer/prompts/corporate_summarizer.tmpl'
        }

        file_path = prompt_files.get(category)
        if not file_path:
            return None

        # Read the base system prompt from the file
        with open(file_path, 'r') as file:
            base_system_prompt = file.read()

        return base_system_prompt

    def find_new_metrics(self, df: pd.DataFrame, df_name: str, category: str) -> list:

        self.base_system_prompt = self.find_new_metrics_prompt(category)
        
        # Formulate the complete system prompt
        prompt = f"""
            {self.base_system_prompt}
         
            Identify the top 3 metrics that can be derived from a given dataset. The metrics must be calculable from each data row. 
            Leverage your domain knowledge, the derived metrics should meaningful and convey important message.
            
            Analyzing the dataset '{df_name}', which includes the following columns: {df.columns}. 
            What are the top 3 metrics that can be derived from this dataset? Please list them without explanation. 

            Respond only with the metric names in the format: ["metric 1", "metric 2", "metric 3"].
        """

        prompt_template = ChatPromptTemplate.from_template(prompt)

        chain = prompt_template | self.model

        return json.loads(chain.invoke({}).content)

    def create_new_col(self, new_metrics: list, df: pd.DataFrame, df_name: str) -> pd.DataFrame:

        system_prompt = self.base_system_prompt + """
            Provide only executable Python code. The code should:
            1. Use 'df' to represent the dataframe.
            2. Not read the dataset into a dataframe.
            3. Not print out any new values.
            4. Add new columns as specified.
            5. Require no further input or interaction.
        """

        message = f"""
            Dataset name: {df_name}
            Column names: {', '.join(df.columns)}
            Required new columns: {', '.join(new_metrics)}

            Generate executable Python code to add these required new columns to 'df'. The code should follow the above guidelines \
            and not perform any additional actions.

            Return only python code in Markdown format, e.g.:

            ```python
            ....
            ```
        """

        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", message)])

        def _sanitize_output(text: str):
            _, after = text.split("```python")
            code = after.split("```")[0]
            return(code)

        chain = prompt | self.model | StrOutputParser() | _sanitize_output
        code = chain.invoke({})

        try:
            exec(code)
        except Exception as e:
            print(f"An error occurred: {e}")

        return df

    def base_summary(self, df: pd.DataFrame, n_samples=3) -> list:
        def get_samples(column):
            non_null = column.dropna().unique()
            return non_null[:n_samples].tolist()

        properties_list = []
        for column in df.columns:
            col_data = df[column]
            dtype = col_data.dtype
            
            # Determine the data type
            if dtype in [int, float, complex]:
                dtype_str = "number"
                min_val = float(col_data.min()) if dtype == float else int(col_data.min())
                max_val = float(col_data.max()) if dtype == float else int(col_data.max())
            elif dtype == bool:
                dtype_str, min_val, max_val = "boolean", None, None
            elif pd.api.types.is_datetime64_any_dtype(col_data) or (dtype == object and pd.to_datetime(col_data, errors='coerce').notna().any()):
                dtype_str, min_val, max_val = "date", col_data.min(), col_data.max()
            elif dtype == object:
                dtype_str, min_val, max_val = "string" if col_data.nunique() / len(col_data) >= 0.5 else "category", None, None
            else:
                dtype_str, min_val, max_val = str(dtype), None, None

            properties = {
                "dtype": dtype_str,
                "min": min_val,
                "max": max_val,
                "num_unique_values": col_data.nunique(),
                "samples": get_samples(col_data),
                "semantic_type": "",
                "description": ""
            }

            properties_list.append({"column": column, "properties": properties})

        return properties_list

    def add_descriptions(self, summary: dict) -> dict:
        system_prompt = """
            You are an experienced data analyst that can annotate datasets. Your instructions are as follows:
            i) Dataset Description: Provide a brief yet informative description of the dataset.
            ii) Field Description: For every data field, craft a concise description.
            iii.) Semantic Type Identification: Assign a precise semantic type to each field based on its values. \
                Use single-word identifiers like 'company', 'city', 'number', 'supplier', 'location', 'gender', \
                'longitude', 'latitude', 'url', 'ip address', 'zip code', 'email', etc.
            Your output should be a neatly updated JSON dictionary without any preamble or explanation.
        
            Annotate the dictionary below. Only return a JSON object.
            
            {{summary}}
        """

        prompt = PromptTemplate.from_template(system_prompt, template_format="jinja2")
        chain = prompt | self.model | SimpleJsonOutputParser()
        return chain.invoke({"summary": summary})

    def summarize(
        self, df: pd.DataFrame,
        df_name: str, 
        category: str,
    ):

        new_metrics = self.find_new_metrics(df, df_name, category)

        print(new_metrics)

        df = self.create_new_col(new_metrics, df, df_name)
        
        base_summary = self.base_summary(df)
        summary = {
            "dataset_description": "",
            "fields": base_summary,
        }

        ## Corporate Financial Dataset consists of too many columns, so GPT cannot easily return the summary with desciption
        if category != "Corporate Financial Dataset":
            summary = self.add_descriptions(summary)

        return summary, df