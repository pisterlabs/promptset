import json

import openai
from openai import OpenAI
import streamlit as st

from src.app.project import Project
from src.utils.session_state_keys import OPENAI_API_KEY, OPENAI_MODEL_NAME


def data_schema_enhancement(project: Project):
    updated_fields = []
    try:
        if project.app.properties[OPENAI_API_KEY]:
            data_schema_dict = project.data_schema.to_dict()
            client = OpenAI(api_key=project.app.properties[OPENAI_API_KEY])
            message = f"""
                        You are a helpful assistant. Your task is to enhance a data schema as I am going to explain.
                        You will receive a JSON object (the data schema) containing, for each column in a dataset called {project.name}, the following
                        attributes:
                        
                        - type: the type of the column (textual, numeric, datetime...)
                        - readable_name: if a column has a weird, ambiguous or very long name, readable_name is used to be displayed replacing the original name. The default value of readable_name is the original name
                        - categories: if the column is categorical, the list of categories (i.e. unique values in the column). Each category has a list of synonyms.
                        - synonyms: Just like the categories, the columns (i.e. their names) can have synonyms
                        
                        Your task is to add more knowledge into this data schema, but only if you find it necessary.
                        You can modify the readable_name of a column if you can think of something better when it is difficult to get the meaning of the column name.
                        The column type and original name (i.e. the JSON keys) cannot be modified.
                        Add new synonyms both for the column names and for each column category, but again, only if you consider them relevant to better understand the data schema. The more synonyms, the better (usually between 0 and 6 is OK). Slight variations of the same value are also allowed.
                        If the column names or categories are not in English, you can add translations as if they were synonyms. Only add synonyms in English.
                        Remember to modify only those elements where you consider it necessary. You can leave some attributes as they are provided if no further knowledge is necessary.
                        
                        As an example, if you get the category "Femenino" ("Female" in Spanish), you can generate the synonyms female, woman, women, F and girl.
                        "City" could have the synonyms region, zone, or municipality.
                        Consider also semantically similar words as synonyms. For example "remuneration" can have the synonyms salary, money and amount of money.
                        If there are values with concatenated words (e.g. AnualGrossSalary or anual_gross_salary), you can generate Anual Gross Salary as a synonym
                        
                        This is the data schema JSON. You must return a JSON answer with the same structure.
                        
                        {data_schema_dict}
                        """
            response = client.chat.completions.create(
                model=project.app.properties[OPENAI_MODEL_NAME],
                messages=[
                    {"role": "user", "content": message}
                ],
                response_format={"type": "json_object"}
            )
            new_data_schema = json.loads(response.choices[0].message.content)
            for field_name, field_data in new_data_schema.items():
                field_updated = False
                if field_name in [field.original_name for field in project.data_schema.field_schemas]:
                    field_schema = project.data_schema.get_field(field_name)
                    if 'readable_name' in field_data and field_schema.readable_name == field_schema.original_name:
                        if field_schema.readable_name != field_data['readable_name']:
                            field_updated = True
                        field_schema.readable_name = field_data['readable_name']
                    if 'synonyms' in field_data:
                        new_synonyms = set(field_schema.synonyms['en'] + field_data['synonyms'])
                        if set(new_synonyms) != set(field_schema.synonyms['en']):
                            field_updated = True
                        field_schema.synonyms['en'] = list(new_synonyms)
                    if 'categories' in field_data:
                        for category_name, category_data in field_data['categories'].items():
                            if category_name in [category.value for category in field_schema.categories]:
                                category = field_schema.get_category(category_name)
                                if 'synonyms' in category_data:
                                    new_synonyms = set(category.synonyms['en'] + category_data['synonyms'])
                                    if set(new_synonyms) != set(category.synonyms['en']):
                                        field_updated = True
                                    category.synonyms['en'] = list(new_synonyms)
                            else:
                                st.error(f'The generated data schema has a non-existent category in {field_name}: {category_name}')
                else:
                    st.error(f'The generated data schema has a non-existent field: {field_name}')
                if field_updated:
                    updated_fields.append(field_name)
        else:
            st.error('You need to set the OpenAI API key (Settings page) to use this feature.')
    except openai.AuthenticationError as e:
        st.error('Please, introduce a valid OpenAI API key to use this feature.')
    except Exception as e:
        st.error(e)
    finally:
        return updated_fields
