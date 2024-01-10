import spacy
import concurrent.futures
from tabulate import tabulate
import traceback
import pandas as pd
import openai
import json
import unicodedata
import time
import re
import numpy as np
import Levenshtein as lev
nlp = spacy.load('en_core_web_sm')


def find_nearest_brands(target_brand, processed_brands, n_brands=25):
    target_brand = ' '.join([token.lemma_ for token in nlp(target_brand)])
    distances = [lev.distance(target_brand, brand) for brand in processed_brands]

    nearest_brand_indices = np.argsort(distances)[:n_brands]
    return [processed_brands[index] for index in nearest_brand_indices]


def remove_non_latin_characters(text):
    normalized_text = unicodedata.normalize('NFKD', text)
    ascii_text = normalized_text.encode('ASCII', 'ignore').decode('utf-8')
    latin_text = ''.join(char for char in ascii_text if char.isascii())
    return latin_text


# Setup OpenAI API
openai.api_key = ''

# Read CSV file
df = pd.read_csv(r'C:\trabalho_upwork\brian_algopix\new_task_23-05\files_from_brian\merged_algopixDetails_algopixAnalysis_marketplaceKingfisher_PM11_VL11_070_brian.csv', sep="|", dtype=str)

responses = []
valid_rows = []
count_rows = 0

list_of_attributes_and_ean = []
master_dict = {}
all_results = []

all_user_messages = []

# replace with nan
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# calculate the number of threads - one for every 20 RPM, but not more than 100
max_workers = min(3500 // 20, 100)


# Iterate over the rows of the dataframe
def process_row(row):

    tokens_used = 0
    type_parameters = []

    # if Details_EAN is empty or nan skip this row
    if pd.isnull(row['Analysis_titles_en']):
        return None

    chat_models = []

    initial_message_for_chatgpt = "Select the attribute's correct value for the product from the value list. Explain your thinking step by step. Please provide the correct word between writing ||. Example: ||Black|| \n\nExplanation:"

    # get attribute names
    # attribute_names = ["Acquisition brand", "Pack Type", "REACh Verified", "Legal information"]
    # lista de chaves para excluir
    excluded_attribute = [
        "Category",
        "Shop SKU",
        "Name",
        "EAN",
        "Main Image 1",
        "Secondary Image 1",
        "Secondary Image 2",
        "Secondary Image 3",
        "Secondary Image 4",
        "Secondary Image 5",
        "Secondary Image 6",
        "Secondary Image 7",
        "Secondary Image 8",
        "Product Guide",
        "Product Instruction Manual",
        "Safety Manual",
        "Video",
        "Unique Selling Point 01",
        "Unique Selling Point 02",
        "Unique Selling Point 03",
        "Unique Selling Point 04",
        "Unique Selling Point 05",
        "Unique Selling Point 06",
        "Unique Selling Point 07",
        "Unique Selling Point 08",
        "Body Copy", "Selling Copy", 'REACh Verified', 'Contains wood and/or paper', 'FSC or PEFC certified', 'Legal information', 'MultiSKU Product Group ID',
        'Key Feature', 'Product diameter', 'Product length', 'Product width', 'Product height', 'Product weight'
    ]

    # assuming that the column containing the json is called 'json_column'
    json_column = json.loads(row['pm_11_results'])

    # get all keys from the json and add them to a list
    attribute_names = [key for key in json_column.keys() if key not in excluded_attribute]

    json_data = row["pm_11_results"]
    json_data = json.loads(json_data)
    attribute_definitions = {}

    for key, value in json_data.items():

        lower_key = key.strip().lower()
        if lower_key in (name.strip().lower() for name in attribute_names):
            # If the key is already in the dictionary, we update the existing dictionary
            if lower_key in attribute_definitions:
                attribute_definitions[lower_key].update({'description': value["description"], 'type_parameter': value["type_parameter"], 'code': value["code"]})
            # Otherwise, start a new dictionary
            else:
                attribute_definitions[lower_key] = {'description': value["description"], 'type_parameter': value["type_parameter"], 'code': value["code"]}

    # get all type_parameters with respective lower_key
    type_parameters.extend([{key: value["code"]} for key, value in attribute_definitions.items()])

    json_object = row["association_list"]
    # decode json string
    json_data = json.loads(json_object)
    values_dict = {}
    for attribute, attribute_info in attribute_definitions.items():
        type_parameter = attribute_info.get('type_parameter', None)
        if type_parameter:
            for item in json_data:
                for key, value in item.items():
                    if key == "json_data_vl11_code":
                        if value.lower() == type_parameter.lower():
                            if attribute not in values_dict:
                                values_dict[attribute] = [item["vl11_value"]]
                            else:
                                values_dict[attribute].append(item["vl11_value"])

    # For each attribute, join its values with '|'
    for attribute, values in values_dict.items():
        values_dict[attribute] = "|".join(values[0])

    # Now merge attribute_definitions and values_dict
    for attribute, attribute_info in attribute_definitions.items():
        attribute_info['values'] = values_dict.get(attribute, [])

    if pd.notnull(row["Analysis_titles_en"]) and row["Analysis_titles_en"] != '':
        product_title = row["Analysis_titles_en"]
    elif pd.notnull(row["CM_data.name"]) and row["CM_data.name"] != '':
        product_title = row["CM_data.name"]
    elif pd.notnull(row["CM_data.Unique Selling Point 01"]) and row["CM_data.Unique Selling Point 01"] != '':
        product_title = row["CM_data.Unique Selling Point 01"]
    else:
        product_title = None

    product_description = row["Details_result_result_additionalAttributes_bulletPoint"]

    keys = [
        "Details_result_result_additionalAttributes_warrantyDescription",
        "Details_result_result_additionalAttributes_modelName",
        "Analysis_model",
        "Details_result_result_additionalAttributes_material",
        "Details_result_result_additionalAttributes_unitCount_type",
        "Details_result_result_additionalAttributes_unitCount_value",
        "Analysis_color"]
    additional_attributes = ""

    prefix = "Details_result_result_additionalAttributes_"

    for key in keys:
        clean_key = key.replace(prefix, "")
        title = clean_key.split("_")[-1]
        if "value" in clean_key:
            title = clean_key.replace("_value", "").split("_")[-1]

        # if the value associated with the key is NaN or None, skip to the next iteration
        if pd.isna(row[key]) or row[key] is None:
            continue

        value = str(row[key])
        additional_attributes += f"{title.capitalize()}: {value}\n"

    list_of_columns = attribute_names
    # columns from merge to add to list_of_columns
    # columns_to_add = ["Analysis_title","EAN","CM_data.category"]
    # list_of_columns.extend(columns_to_add)
    # insert analysis_title in list_of_columns
    list_of_columns.insert(0, "Analysis_title")
    # insert EAN in list_of_columns
    list_of_columns.insert(0, "EAN")

    for attribute_name, attribute_definition in attribute_definitions.items():
        # if values is empty skip this attribute
        if len(attribute_definition['values']) == 0:
            continue

        # if is acquisition brand we have a lot of values and we need to get the nearest brand
        if "acquisition brand" == attribute_name:
            # need to check gpt to get the brand from Analysis_titles_en and Analysis_brands
            analysis_titles_en = row["Analysis_titles_en"]
            analysis_brand = row['Analysis_brand']

            if analysis_titles_en and analysis_brand and type(analysis_titles_en) == str and type(analysis_brand) == str:
                while True:
                    try:
                        # send request to gpt to get brand
                        response_brand = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "I have a product title and brand below. Please state the brand name. Do not include words like 'Enterprises', 'Tools', 'Company' etc. in the brand name. Always provide the correct word (brand) between writing ||. Example: ||Black||"},
                                {"role": "user", "content": f"\nproduct title: {analysis_titles_en} sold by {analysis_brand}"}
                            ]
                        )

                        for choice in response_brand.choices:
                            messages_from_response = choice.message
                            if not isinstance(messages_from_response, list):
                                messages_from_response = [messages_from_response]

                            for messages_from_response_2 in messages_from_response:
                                all_user_messages.append(
                                    {
                                        'ean': row["Details_Search Term"],
                                        'prompt':
                                            [
                                                {"role": "system", "content": "I have a product title and brand below. Please state the brand name. Do not include words like 'Enterprises', 'Tools', 'Company' etc. in the brand name. Always provide the correct word (brand) between writing ||. Example: ||Black||"},
                                                {"role": "user", "content": f"\nproduct title: {analysis_titles_en} sold by {analysis_brand}"}
                                        ],
                                        'response': messages_from_response_2.content,
                                        'final_response': (lambda x: x.group(0).replace("|", "") if x else None)(re.search(r'\|\|(.*?)\|\|', messages_from_response_2.content))
                                    }
                                )

                        tokens_used = tokens_used + int(response_brand['usage']['total_tokens'])

                        response_brand = str(response_brand.choices[0].message['content']).strip()

                        if not re.search(r'\|\|.*\|\|', response_brand):
                            response_brand = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "I have a product title and brand below. Please state the brand name. Do not include words like 'Enterprises', 'Tools', 'Company' etc. in the brand name. Always provide the correct word (brand) between writing ||. Example: ||Black||"},
                                    {"role": "user", "content": f"\nproduct title: {analysis_titles_en} sold by {analysis_brand}"},
                                    {"role": "user", "content": response_brand},
                                    {"role": "assistant", "content": response_brand},
                                    {"role": "user", "content": "There is no word between '||' in the content. Please provide the correct answer between the characters '||'. "}
                                ]
                            )

                            for choice in response_brand.choices:
                                messages_from_response = choice.message
                                if not isinstance(messages_from_response, list):
                                    messages_from_response = [messages_from_response]

                                for messages_from_response_2 in messages_from_response:
                                    all_user_messages.append(
                                        {
                                            'ean': row["Details_Search Term"],
                                            'prompt':
                                                [
                                                    {"role": "system", "content": "I have a product title and brand below. Please state the brand name. Do not include words like 'Enterprises', 'Tools', 'Company' etc. in the brand name. Always provide the correct word (brand) between writing ||. Example: ||Black||"},
                                                    {"role": "user", "content": f"\nproduct title: {analysis_titles_en} sold by {analysis_brand}"},
                                                    {"role": "user", "content": response_brand},
                                                    {"role": "assistant", "content": response_brand},
                                                    {"role": "user", "content": "There is no word between '||' in the content. Please provide the correct answer between the characters '||'. "}
                                            ],
                                            'response': messages_from_response_2.content,
                                            'final_response': (lambda x: x.group(0).replace("|", "") if x else None)(re.search(r'\|\|(.*?)\|\|', messages_from_response_2.content))
                                        }
                                    )

                            tokens_used = tokens_used + int(response_brand['usage']['total_tokens'])

                            response_brand = str(response_brand.choices[0].message['content']).strip()

                        if "undefined" in response_brand.lower() or "different" in response_brand.lower() or "nan" in response_brand.lower() or response_brand == "" or response_brand is None:
                            response_brand = f"||{analysis_brand}||"

                        brands_values = attribute_definition['values'].split("|")

                        word_inside_brackets = re.search(r'\|\|(.*?)\|\|', response_brand)
                        if word_inside_brackets is not None:
                            word_inside_brackets = word_inside_brackets.group(0)
                        else:
                            # Handle the case when no match is found
                            word_inside_brackets = ''

                        # check if the response_brand is in the brands_values list
                        if word_inside_brackets.lower().replace("||", "").strip() in [brand.lower() for brand in brands_values]:
                            final_brand = word_inside_brackets
                            responses.append(final_brand)
                        else:
                            # lemmatizer = WordNetLemmatizer()
                            # lemmatizer = nlp

                            # Tokenize and lemmatize each brand in the list
                            # processed_brands = [' '.join([lemmatizer.lemmatize(token) for token in word_tokenize(brand)]) for brand in brands_values]
                            processed_brands = [' '.join([token.lemma_ for token in nlp(brand)]) for brand in brands_values]

                            # Test with a specific brand
                            top_20_brands = find_nearest_brands(word_inside_brackets, processed_brands)

                            # start_sequence = "|"
                            response_brand = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "assistant", "content": response_brand},
                                    {"role": "user", "content": f"product title: {analysis_titles_en} sold by {analysis_brand}"},
                                    {"role": "user", "content": "I have a list of 25 brands. PLease try to identify the correct brand for the product in the list. Write the brand surrounded by ||  ||. If the brand is not in the list, write null. \nExample ||Brand||\n\n"},
                                    {"role": "user", "content": f"Brands: {top_20_brands}\n\n"}
                                ]
                            )

                            for choice in response_brand.choices:
                                messages_from_response = choice.message
                                if not isinstance(messages_from_response, list):
                                    messages_from_response = [messages_from_response]

                                for messages_from_response_2 in messages_from_response:
                                    all_user_messages.append(
                                        {
                                            'ean': row["Details_Search Term"],
                                            'prompt':
                                                [
                                                    {"role": "assistant", "content": response_brand},
                                                    {"role": "user", "content": f"product title: {analysis_titles_en} sold by {analysis_brand}"},
                                                    {"role": "user", "content": "I have a list of 25 brands. PLease try to identify the correct brand for the product in the list. Write the brand surrounded by ||  ||. If the brand is not in the list, write null. \nExample ||Brand||\n\n"},
                                                    {"role": "user", "content": f"Brands: {top_20_brands}\n\n"}
                                            ],
                                            'response': messages_from_response_2.content,
                                            'final_response': (lambda x: x.group(0).replace("|", "") if x else None)(re.search(r'\|\|(.*?)\|\|', messages_from_response_2.content))
                                        }
                                    )

                            tokens_used = tokens_used + int(response_brand['usage']['total_tokens'])

                            response_brand = str(response_brand.choices[0].message['content']).strip()

                            if "undefined" in response_brand.lower() or "different" in response_brand.lower() or "nan" in response_brand.lower() or response_brand == "" or response_brand is None:
                                response_brand = f"||{analysis_brand}||"

                            # get word just inside || from response_brand
                            word_inside_brackets = re.search(r'\|\|(.*?)\|\|', response_brand)
                            if word_inside_brackets is not None:
                                word_inside_brackets = word_inside_brackets.group(0)
                            else:
                                # Handle the case when no match is found
                                word_inside_brackets = ''

                            if word_inside_brackets.lower().replace("||", "").strip() in [brand.lower() for brand in brands_values]:
                                # print(response_brand)
                                # print("Found in brands_values with 20 selected words")
                                # if exists in the list its final_brand will be the brand
                                final_brand = f"{word_inside_brackets}"
                                responses.append(final_brand)
                            else:
                                # if "undefined" in response_brand.lower() or "different" in response_brand.lower() or "nan" in response_brand.lower() or response_brand == "" or response_brand is None:
                                # print("Invalid response from chatgpt")
                                # print("Not found in brands_values with 20 selected words")
                                response_brand = f"||{analysis_brand}||"
                                responses.append(response_brand)
                        break
                    except Exception as e:
                        print("An error occurred:", str(e))
                        traceback.print_exc()  # Isto imprimirá o traceback completo.
                        time.sleep(10)  # Sleep for 10 seconds before retrying

            # User message without line breaks
            user_message = f"Attribute Name: {attribute_name}\n" \
                f"Attribute Definition: {attribute_definition['description']}\n" \
                f"Product: {product_title}\n" \
                f"Product Description: {product_description}\n" \
                f"Additional Attributes: {additional_attributes}\n\n"\
                f"{initial_message_for_chatgpt}"

            # Store user_message to a list
            valid_rows.append(user_message)

            ean = row["Details_Search Term"]

            if ean not in master_dict:
                master_dict[ean] = {"EAN": ean, "Name": product_title, "category": row['CM_data.category'], "Product Weight": row['Analysis_weight_item_dimensions'],
                                    "Product Length": row['Analysis_length'], "Product Width": row['Analysis_width'], "Product Height": row['Analysis_height']}

            attribute_dict = master_dict[ean]

            for column in list_of_columns:
                if column in ("EAN", "Name"):  # these columns have already been handled
                    continue
                if column.strip().lower() in attribute_name:
                    if column in attribute_dict:
                        continue
                    attribute_dict[column] = response_brand.split('||')[-2].strip() if response_brand.count('||') >= 2 else None
                    # list of values to replace
                    to_replace = [
                        'no information', 'None', '', 'nan', 'Nan', 'NaN', 'n/a', 'N/A', 'unknown', 'Unknown', 'UNKNOWN',
                        'not available', 'Not Available', 'NOT AVAILABLE',
                        'undetermined', 'Undetermined', 'UNDETERMINED',
                        'cannot be determined', 'Cannot be determined', 'CANNOT BE DETERMINED',
                        'not applicable', 'Not Applicable', 'NOT APPLICABLE',
                        'not determinable', 'Not determinable', 'NOT DETERMINABLE',
                        'missing', 'Missing', 'MISSING',
                        'not specified', 'Not Specified', 'NOT SPECIFIED',
                        '--', '---', 'na', 'NA', 'none', 'None', 'NONE',
                        'null', 'Null', 'NULL', ''
                    ]

                    # check if attribute_dict[column] is not None before trying to lower
                    if attribute_dict[column] is not None:
                        attribute_dict[column] = attribute_dict[column].lower()

                        # Join elements in to_replace with | for regex
                        to_replace = '|'.join(r'\b{}\b'.format(x.lower()) for x in to_replace)

                        # Use regex sub to replace exact matches
                        attribute_dict[column] = re.sub(to_replace, '', attribute_dict[column])
                    else:
                        attribute_dict[column] = ''

            continue

        # User message without line breaks
        user_message = f"Attribute Name: {attribute_name}\n" \
            f"Attribute Definition: {attribute_definition['description']}\n" \
            f"Value List: {attribute_definition['values']}\n" \
            f"Product: {row['Analysis_brand']} {product_title}\n" \
            f"Product Description: {product_description}\n" \
            f"Additional Attributes: {additional_attributes}\n\n"\
            f"{initial_message_for_chatgpt}"

        # Example usage
        # latin_text = remove_non_latin_characters(user_message)

        # if attribute_definition['values'] is empty continue
        if len(attribute_definition['values']) == 0:
            # store response
            responses.append("No value list for this attribute")

            # Store user_message to a list
            valid_rows.append(user_message)

            continue

        while True:
            try:
                # Create chat models with the row message
                chat_models = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a Machine Learning Program building an attribute from product data."},
                        {"role": "user", "content": user_message}
                    ]
                )

                for choice in chat_models.choices:
                    messages_from_response = choice.message
                    if not isinstance(messages_from_response, list):
                        messages_from_response = [messages_from_response]

                    for messages_from_response_2 in messages_from_response:
                        all_user_messages.append(
                            {
                                'ean': row["Details_Search Term"],
                                'prompt':
                                    [
                                        {"role": "system", "content": "You are a Machine Learning Program building an attribute from product data."},
                                        {"role": "user", "content": user_message}
                                ],
                                'response': messages_from_response_2.content,
                                'final_response': (lambda x: x.group(0).replace("|", "") if x else None)(re.search(r'\|\|(.*?)\|\|', messages_from_response_2.content))
                            }
                        )

                tokens_used = tokens_used + int(chat_models['usage']['total_tokens'])

                content = str(chat_models.choices[0].message['content']).strip()

                if not re.search(r'\|\|.*\|\|', content):
                    chat_models = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "assistant", "content": content},
                            {"role": "system", "content": "You are a Machine Learning Program building an attribute from product data."},
                            {"role": "user", "content": user_message},
                            {"role": "user", "content": f"There is no word between '||' in the content. Please provide the correct answer between the characters '||' for the {attribute_name}: "}
                        ]
                    )

                    for choice in chat_models.choices:
                        messages_from_response = choice.message
                        if not isinstance(messages_from_response, list):
                            messages_from_response = [messages_from_response]

                        for messages_from_response_2 in messages_from_response:
                            all_user_messages.append(
                                {
                                    'ean': row["Details_Search Term"],
                                    'prompt':
                                        [
                                            {"role": "assistant", "content": content},
                                            {"role": "system", "content": "You are a Machine Learning Program building an attribute from product data."},
                                            {"role": "user", "content": user_message},
                                            {"role": "user", "content": f"There is no word between '||' in the content. Please provide the correct answer between the characters '||' for the {attribute_name}: "}
                                    ],
                                    'response': messages_from_response_2.content,
                                    'final_response': (lambda x: x.group(0).replace("|", "") if x else None)(re.search(r'\|\|(.*?)\|\|', messages_from_response_2.content))
                                }
                            )

                    tokens_used = tokens_used + int(chat_models['usage']['total_tokens'])

                break
            except Exception as e:
                print("An error occurred:", str(e))
                traceback.print_exc()  # Isto imprimirá o traceback completo.
                time.sleep(10)  # Sleep for 10 seconds before retrying

        response = str(chat_models.choices[0].message['content']).strip()

        responses.append(response)

        # Store user_message to a list
        valid_rows.append(user_message)

        ean = row["Details_Search Term"]

        if ean not in master_dict:
            master_dict[ean] = {"EAN": ean, "Name": product_title, "category": row['CM_data.category'], "Product Weight": row['Analysis_weight_item_dimensions'],
                                "Product Length": row['Analysis_length'], "Product Width": row['Analysis_width'], "Product Height": row['Analysis_height']}

        attribute_dict = master_dict[ean]

        for column in list_of_columns:
            if column in ("EAN", "Name"):  # these columns have already been handled
                continue
            if column.strip().lower() in attribute_name:
                if column in attribute_dict:
                    continue
                attribute_dict[column] = response.split('||')[-2].strip() if response.count('||') >= 2 else None
                # list of values to replace
                to_replace = [
                    'no information', 'None', '', 'nan', 'Nan', 'NaN', 'n/a', 'N/A', 'unknown', 'Unknown', 'UNKNOWN',
                    'not available', 'Not Available', 'NOT AVAILABLE',
                    'undetermined', 'Undetermined', 'UNDETERMINED',
                    'cannot be determined', 'Cannot be determined', 'CANNOT BE DETERMINED',
                    'not applicable', 'Not Applicable', 'NOT APPLICABLE',
                    'not determinable', 'Not determinable', 'NOT DETERMINABLE',
                    'missing', 'Missing', 'MISSING',
                    'not specified', 'Not Specified', 'NOT SPECIFIED',
                    '--', '---', 'na', 'NA', 'none', 'None', 'NONE',
                    'null', 'Null', 'NULL', ''
                ]

                # check if attribute_dict[column] is not None before trying to lower
                if attribute_dict[column] is not None:
                    attribute_dict[column] = attribute_dict[column].lower()

                    # Join elements in to_replace with | for regex
                    to_replace = '|'.join(r'\b{}\b'.format(x.lower()) for x in to_replace)

                    # Use regex sub to replace exact matches
                    attribute_dict[column] = re.sub(to_replace, '', attribute_dict[column])
                else:
                    attribute_dict[column] = ''

    # Converting all_results into a pandas DataFrame
    all_results = list(master_dict.values())

    # Converting all_results into a pandas DataFrame
    df_final = pd.DataFrame(all_results)

    # remove duplicates from type_parameters
    unique_type_parameters = []
    for d in type_parameters:
        if d not in unique_type_parameters:
            unique_type_parameters.append(d)

    type_parameters = unique_type_parameters

    # Create an empty dictionary to store code
    parameters_dict = {}

    # Loop through the dictionaries in type_parameters
    for parameter in type_parameters:
        # Loop through the columns of the DataFrame
        for column in df_final.columns:
            # Check if the dictionary key matches the column name
            if column.lower() == list(parameter.keys())[0].lower():
                # Add the corresponding value in the dictionary
                parameters_dict[column] = parameter[list(parameter.keys())[0]]
                break

    # Create DataFrame with code
    df_codes = pd.DataFrame([parameters_dict], columns=df_final.columns)

    # Concatenate the original DataFrame with the codes
    df_final = pd.concat([df_codes, df_final], ignore_index=True)

    # Save the dataframe to a CSV file
    df_final.to_csv('save_response_from_chatgpt_100.csv', index=False, sep="|")

    # return tokens_used, df_final
    return tokens_used, df_final


# List to store all the final dataframes returned from process_row.
all_df_finals = []
# Initialize the token counter.
total_tokens = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(process_row, row): row for _, row in df.head(3).iterrows()}

    # Collect results as they become available and update the token counter.
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result is not None:
            tokens, df_final = result  # Retrieve the used tokens from the result.
            total_tokens += tokens  # Update the token counter.
            if df_final is not None:
                all_df_finals.append(df_final)  # Append the dataframe to the list.
            else:
                # Handle the None result
                print("Empty result for a row in the DataFrame")
        else:
            # Handle the None result
            print("Empty result for a row in the DataFrame")

# Combine all the dataframes into a single dataframe
df_final = pd.concat(all_df_finals)

# summary
# Calculate line difference between merge_spreadsheet and df_final
difference_merge_and_result = (df.shape[0] - (df_final.shape[0] - 1))

# Making a copy of the DataFrame
df_final_copy = df_final.copy()

# Replacing blank spaces with NaN in the DataFrame copy
df_final_copy.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Disregarding the first row in the DataFrame copy
df_final_copy = df_final_copy.iloc[1:]

# Counting the number of non-empty cells (i.e., with some value) in each row and summing
all_attributes_not_empty = df_final_copy.apply(lambda row: row.count(), axis=1).sum()

# average attributes per item
average_not_empty = all_attributes_not_empty / (df_final.shape[0] - 1)

# average tokens per EAN
average_tokens_per_EAN = total_tokens / (df_final.shape[0] - 1)
# Define table data
table_data = [
    ["Items with no Name - Difference Merge/Final", difference_merge_and_result],
    ["Total of attributes displayed", all_attributes_not_empty],
    ["Average attributes per item", average_not_empty],
    ["Total of EANs", df_final.shape[0] - 1],
    ["Total of tokens", total_tokens],
    ["Average of tokens per EAN", average_tokens_per_EAN],
]

# print the table
print("\n")
print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='pipe'))

# list of values to replace before save all_user_messages to a CSV file
to_replace = [
    'no information', 'None', '', 'nan', 'Nan', 'NaN', 'n/a', 'N/A', 'unknown', 'Unknown', 'UNKNOWN',
    'not available', 'Not Available', 'NOT AVAILABLE',
    'undetermined', 'Undetermined', 'UNDETERMINED',
    'cannot be determined', 'Cannot be determined', 'CANNOT BE DETERMINED',
    'not applicable', 'Not Applicable', 'NOT APPLICABLE',
    'not determinable', 'Not determinable', 'NOT DETERMINABLE',
    'missing', 'Missing', 'MISSING',
    'not specified', 'Not Specified', 'NOT SPECIFIED',
    '--', '---', 'na', 'NA', 'none', 'None', 'NONE',
    'null', 'Null', 'NULL', ''
]

# save all_user_messages to a CSV file
all_user_messages_df = pd.DataFrame(all_user_messages)
all_user_messages_df['final_response'] = all_user_messages_df['final_response'].replace(to_replace, np.nan)
all_user_messages_df.to_csv('all_user_messages_100.csv', index=False, sep="|")
