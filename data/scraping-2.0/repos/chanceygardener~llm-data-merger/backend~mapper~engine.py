import pandas as pd
from langchain.chains import SequentialChain, LLMChain, TransformChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

import re
import json
import asyncio
import os
from typing import List, Optional, Dict
from schema import TableMeta, TableColumnMapping
from io import StringIO



fpa = '/Users/chanceygardener/interviews/zero/task/table_A.csv'
fpb = '/Users/chanceygardener/interviews/zero/task/table_B.csv'
tmpth = '/Users/chanceygardener/interviews/zero/task/template.csv'

clean_template = '''Given the following CSV file:
{csv}
# END CSV

return a cleaned version where columns with values inconsistent with their titles are removed and values are quoted to avoid misaligned columns when reading the file.

CSV:
'''

clean_prompt = PromptTemplate(input_variables=['csv'], template=clean_template)



llm = OpenAI(model_name='text-davinci-003', temperature=0, max_tokens=-1)
clean_chain = LLMChain(prompt=clean_prompt, llm=llm)
# "format_template": "a python format string template in the format of the data value. The variables in the string template should correspond to the named capture groups in the 'regex' field.",
desc_template = '''TABLE:
{table}
####### END TABLE
{additional_context}
Generate a JSON mapping columns from the table to a structure summarizing the column, use the following format:
{{
    "description": "A concise summary of the overall role of the table",
    "columns": {{
        "<table_column_name_1>": {{
            "description": "A detailed description of the column's role in the context of the table",
            "python_type": "The python type of the column's data values, use datetime.datetime for all date values",
            "regex": "a \\ escaped python regular expression with matching the format of the column's data values. The regex must have named capturing groups to parse the value into any identifiable components such as prefixes, decimal places, first and last names, etc. Each capturing group must have a meaningful name",
            "sample_values": ["example_value_1", "example_value_2", "example_value_3"]
        }},
        ...
        "<table_column_n>": {{
            ...
        }}
    }}
}}

JSON:'''

desc_template = PromptTemplate(
    template=desc_template, input_variables=['table', 'additional_context'])
desc_chain = LLMChain(llm=llm, prompt=desc_template, verbose=True)

# For each key in the TEMPLATE object, identify the following:
# 1. The column name from the INPUT object most likely to be the analog of the TEMPLATE object key.
# 2. Reasoning explaining why the two fields are similar.

# If there is ambiguity between the candidacy of multiple values, the value mapped to the TEMPLATE object key should be an array sorted from most to least likely. Even if there is only one likely value, return an array of length 1.


map_template = '''TEMPLATE_OBJECT: {template_obj}

INPUT_OBJECT: {input_obj}

Return a JSON object that maps the fields featured in the template object to all possible analogous fields from the input object sorted in descending order from the most to least likely to be equivalent.

Use the following format:
{{
    "template_object_key_1": [
            {{
                "input_obj_key": "The key from the 'INPUT' object corresponding to the column that is a candidate for equivalency to 'template_object_key_1'",
                "reason": "A detailed explanation of why this 'input_obj_key' is a likely analog to 'template_object_key_1'",
                "translation_format": "a python format string template with variables corresponding to the 'components' field of INPUT_OBJECT['<input_obj_key>']. The template string maps these variables into the format represented by the regex field in the template object at TEMPLATE_OBJECT['<template_object_key_1>']. Any wrapping quote characters should be removed. Example: values matching the regex '^(?P<prefix>[A-Z]{{2}})(?P<number>\\d{{5}})$' would have a translation format of '{{prefix}}{{number}}' "
            }}
        ],
    ...
    "template_object_key_n": [...]
}}

JSON:'''
# to produce a string value of the format represented in TEMPLATE[<template_object_key_1>]["sample_values"] . The template should include variables for each value in TEMPLATE[<template_object_key_1>]["components"]
map_prompt = PromptTemplate(template=map_template, input_variables=[
                            'template_obj', 'input_obj'])
map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=True)

get_format_template = '''SOURCE FORMAT: {target_format}

INPUT VALUES:
{input_values}


Generate a JSON object according to the following specification:
{{
    "source_regex": "A python regular expression with named capture groups representing the significant fields",
    "target_fmt": "a python format string in the format of the  with variables corresponding to the named capture groups in the source regex"
}}

Ensure that the named capture groups in the source regex are the same as those in the target regex.
JSON:'''

get_format_prompt = PromptTemplate(template=get_format_template, input_variables=[
                                   'input_values', 'target_format'])
get_format_chain = LLMChain(prompt=get_format_prompt, llm=llm)


def get_sample_string(df):
    return df.head(10).to_csv()

with open(fpa) as ifile:
    table_a = ifile.read()
with open(fpb) as ifile:
    table_b = ifile.read()

template = pd.read_csv(tmpth)
# begin main

# 1st get template descriptions


class TableDescription:
    '''Manages data, descriptions, and formats
    provided by a "template" csv file.'''

    @staticmethod
    def _format_additional_context(context: Optional[str]) -> str:
        if context is None:
            return ''
        assert isinstance(
            context, str), f'additional context should be string or none, got: {context.__class__.__name}'
        insert_template = '\nConsider the following additional context in your descriptions of the table and its columns: {context} \n'
        return insert_template.format(context=context)

    def get_meta(self):
        print('running get_meta')
        chain_output = desc_chain.run(
            table=self._df.head(10).to_csv(),
            additional_context=self._format_additional_context(
                self.context))
        print("GET META OUTPUT:")
        print(chain_output)

        return TableMeta(**json.loads(chain_output))
    
    def process_csv(self, csv_string) -> pd.DataFrame:
        print(f'Raw CSV:\n{csv_string}\n## END RAW CSV\n')
        chain_output = clean_chain.run(csv=csv_string)
        print(f'Cleaned CSV:\n{csv_string}\n## END CLEANED CSV\n')
        df = pd.read_csv(StringIO(chain_output))
        print(f'Resulting Dataframe:\n{df}\n')
        return df

    
    def __init__(self, csv_string: str, context=None):
        self._df: pd.DataFrame = self.process_csv(csv_string)
        self._context: str = self._format_additional_context(context)
        self._meta: TableMeta = self.get_meta()

    @property
    def context(self) -> Optional[str]:
        return self._context

    @property
    def meta(self) -> TableMeta:
        return self._meta

    @property
    def table(self) -> pd.DataFrame:
        return self._df


class TableAggregator:

    @staticmethod
    def _get_json_struct_from_chain(chain, prompt_params: dict):
        chain_output = chain.run(**prompt_params)
        print(chain_output)
        return json.loads(chain_output)

    def _get_descriptions(self, df):
        return self._get_table_prompt_val(
            df)

    def _map_to_template(self, input_description):
        map_chain_input_obj = input_description.meta.mapping_description()
        map_chain_template_obj = self._base_template.meta.mapping_description()
        chain_output = map_chain.run(input_obj=map_chain_input_obj,
                                     template_obj=map_chain_template_obj)
        print(f'MAP CHAIN OUTPUT: {chain_output}')
        chain_output = json.loads(chain_output)
        mappings = {
            k: [TableColumnMapping(**chain_output[k][i])
                for i in range(len(chain_output[k]))] for k in chain_output
        }
        return mappings

    def __init__(self, template_df):
        print('creating base template')
        self._base_template = TableDescription(template_df)
        self._aggregated = pd.DataFrame(columns=list(template.columns))
        self._inputs: List[TableDescription] = []

    def _get_format(self, source_values: List[str], target_values: List[str]):
        structs = get_format_chain.run(
            source_values=source_values, target_values=target_values)
        print(structs)
        return structs

    def get_preliminary_mappings(self, input_df):
        print(f'Getting preliminary mappings')
        input_description = TableDescription(input_df)
        prelim_mappings = self._map_to_template(input_description)
        return input_description, prelim_mappings

    def _map_row(self, input_row, final_mappings, table):
        out = {}
        for template_key, mapping in final_mappings.items():
            input_key = mapping.input_obj_key
            input_col_meta = table.meta.columns[input_key]
            input_val = input_row[input_key]
            print(f'Checking {input_key}: {input_val} to match {input_col_meta.regex}')
            print(f'\t the template colum ({template_key}) has a regex value: {self._base_template.meta.columns[template_key].regex}')
            input_parse = input_col_meta.regex.match(str(input_val))
            if not input_parse:
                print(
                    f'\t- Non matching value "{str(input_val)}" for input column {input_key} with pattern r"{input_col_meta.regex.pattern}" - cannot map to {template_key}')
                return
            print(f'\tParsed {input_key} value "{input_val}" with match {input_parse} ({input_parse.re.pattern}) ({input_parse.groups()}) - Groups: {input_parse.groupdict()}')
            # value = self._base_template.meta.columns[template_key].format_value(
            #     **input_parse.groupdict())
            components = input_parse.groupdict()
            print(f'\t\tParsed the following groups: {components} - mapping to {mapping.translation_format}')
            value = mapping.translation_format.format(**components)
            out[template_key] = value
        return out

    def aggregate_mappings(self, table: TableDescription, final_mappings: Dict[str, TableColumnMapping]) -> None:
        # map_cols = self._get_column_mapper(table, final_mappings)
        for i, row in table.table.iterrows():
            new_row = self._map_row(row, final_mappings, table)
            if new_row is None:
                # TODO: handle these anomalies
                print(f'Row {i} did not match format: {row.to_dict()}')
                continue
            self._aggregated = pd.concat([self._aggregated, pd.DataFrame([new_row])], ignore_index=True)
    
    def get_output(self) -> pd.DataFrame:
        return self._aggregated


test = TableAggregator(template)

table_a, mappings_a = test.get_preliminary_mappings(table_a)
table_b, mappings_b = test.get_preliminary_mappings(table_b)

# This is a standin for user validation -- just pick the 1st mapping for an initial test
final_mappings_a = {k: mappings_a[k][0] for k in mappings_a}

select_b = {
    'Premium': 1,
    'PolicyNumber': 1
}
final_mappings_b = {k: mappings_b[k][select_b.get(k, 0)] for k in mappings_b}
# try:
print("\nTABLE A MAPPINGS")
test.aggregate_mappings(table_a, final_mappings_a)
print("\nTABLE B MAPPINGS")
test.aggregate_mappings(table_b, final_mappings_b)
out = test.get_output()
# except Exception as e:
#     print(f'getting final mappings failed with {e.__class__.__name__}: {e}')





# maps = test.get_preliminary_mappings(table_a)
# for template_col, mapping in maps.items():
#     fmt_transform = test.get_format_transform(template_col, mapping)


# print(maps)
