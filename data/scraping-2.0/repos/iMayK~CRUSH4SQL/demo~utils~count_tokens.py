#%%
import openai
import tiktoken

#%%
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


#%%
example_messages = [
    {
        "role": "system",
        "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "New synergies will help drive top-line growth.",
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Things working well together will increase revenue.",
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Let's talk later when we're less busy about how to do better.",
    },
    {
        "role": "user",
        "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
    },
]

#%%
for model in [
    "gpt-3.5-turbo-16k-0613"
    ]:
    print(model)
    # example token count from the function defined above
    print(f"{num_tokens_from_messages(example_messages, model)} prompt tokens counted by num_tokens_from_messages().")

    # example token count from the OpenAI API
    #deployment_name = 'chatgpt-35'
    #response = openai.ChatCompletion.create(
    #    engine = deployment_name,
    #    messages=example_messages,
    #    temperature=0
    #    max_tokens=1,  # we're only counting input tokens here, so let's not waste tokens on the output
    #)
    #print(f'{response["usage"]["prompt_tokens"]} prompt tokens counted by the OpenAI API.')
    #print()


# %%
example_messages_2 = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "Your task is to create a syntactically correct SQL query based on an input question. Make sure to use only the column names specified in the schema description and avoid querying for columns that do not exist. Also, pay attention to which column belongs to which table."
    },
    {
        "role": "assistant",
        "content": "Sure, I'd be happy to!"
    },
    {
        "role": "user",
        "content": "Please follow these instructions carefully:\n First, I will provide you with an example that includes a database schema, a question based on it, and a detailed step-by-step approach on how to construct the SQL query for the question. Pay close attention to the example.\n After the example, I will give you a new question along with its corresponding schema. You will need to write an SQL query for it using the strategy described in the example."
    },
    {
        "role": "assistant",
        "content": "Sure. Please provide the example with the schema information, question, and detailed solution, and I'll assist you accordingly."
    },
    {
        "role": "user",
        "content": "Again, be attentive to the instructions mentioned previously.\n I am providing you the Schema information for the example followed by the corresponding question and the detailed solution.\n Schema information:\n Here are a few rows from each table in the schema. Keep in mind that the tables in the schema may have more rows than what is shown here.\n Table: 'India\’s Imports or Exports of Principal Agricultural Commodities 6165'\n 'Country' | 'Imports or exports' | 'SourceYear' | 'Value of imports or exports of principal agricultural commodities' | 'Quantity of imports or exports of principal agricultural commodities' | 'YearCode' | 'Year' | 'Commodity type'\n 'India' | 'Exports' | '2009-10' | '2973.19' | '2881.22' | '2009' | 'Financial Year (Apr - Mar), 2009' | 'Other cereals'\n 'India' | 'Exports' | '2017-18' | '4913.28' | '714.0' | '2017' | 'Financial Year (Apr - Mar), 2017' | 'Fresh fruits'\n 'India' | 'Imports' | '2012-13' | '3.59' | 'nan' | '2012' | 'Financial Year (Apr - Mar), 2012' | 'Percentage Share of Agricultural Imports in National Imports'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'\n 'Exports in rupees' | 'Country' | 'Commodity name' | 'Export of selected new item commodities to principal countries.' | 'Yearcode' | 'Year'\n '165' | 'India' | 'Carpet' | 'Italy' | '2020' | 'Financial Year (Apr - Mar), 2020'\n '120' | 'India' | 'Tea' | 'Iraq' | '2020' | 'Financial Year (Apr - Mar), 2020'\n '66' | 'India' | 'Cashew' | 'France' | '2019' | 'Financial Year (Apr - Mar), 2019'\n\n Table: 'Selected Economic and Social Indicator 6159'\n 'Exports in crores' | 'Exports in US millions' | 'Country' | 'YearCode' | 'Year'\n '203571' | '44560' | 'India' | '2000' | 'Financial Year (Apr - Mar), 2000'\n '2219854' | '313361' | 'India' | '2019' | 'Financial Year (Apr - Mar), 2019'\n '2159043' | '291808' | 'India' | '2020' | 'Financial Year (Apr - Mar), 2020'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'\n 'Commodity name' | 'Sourceyear' | 'Exports in usd' | 'Exports in rupees' | 'Country' | 'Export to country' | 'Yearcode' | 'Year'\n 'Tobacco' | '1996-97' | '4.2' | '30' | 'India' | 'U.S.A' | '1996' | 'Financial Year (Apr - Mar), 1996'\n 'Carpets' | '2012-13' | '91.0' | '700' | 'India' | 'U.K.' | '2012' | 'Financial Year (Apr - Mar), 2012'\n 'Rice' | '1996-97' | '33.6' | '220' | 'India' | 'Kuwait' | '1996' | 'Financial Year (Apr - Mar), 1996'\n\n\n Question: 'Year wise list of India's export destination countries.'\n\n Solution:\n Step 1: Firstly, let's analyze the tables in the schema.\n\n Table: 'India\’s Imports or Exports of Principal Agricultural Commodities 6165'\n This table contains information about India\'s imports and exports of principal agricultural commodities. However, the question does not specifically ask for agricultural commodities. The 'Country' column contains 'India' as the value, which is not a destination country, but rather the source country. Further, columns such as 'Imports or exports', 'Quantity of imports or exports of principal agricultural commodities', 'Value of imports or exports of principal agricultural commodities', etc. do not mention any information about the destination country. Therefore, for the given question, which is about India\'s export destination countries, this table is not directly useful.\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'\n This table focuses on the export of select commodities from India to different countries and is potentially relevant to the question. Let\'s look into the table content. The column 'Exports in rupees' contains numerical data. Based on the name of the column and values in it, I think the column is about the total amount of the commodity exported in rupees. The 'Country' column contains 'India' as the value, which is not a destination country, but rather the source country. The column 'Commodity name' contains 'Carpet', 'Tea', etc., so this column indicates the type of commodity exported. Column 'Export of selected new item commodities to principal countries.' contains values like 'Italy', 'Iraq', etc. which are basically different countries, so this column provides information about the destination country to which items are exported from India. The 'Year' and 'Yearcode' columns represent the financial year and the corresponding year of the export. This table can provide insights into India's export destination countries and can be utilized to answer the question effectively.\n\n Table: 'Selected Economic and Social Indicator 6159'\n This table contains selected economic and social indicators for India. It is difficult to determine just from the name of the table whether it is relevant or not. Let's analyze the content of the table to get better understanding. The column 'Exports in crores' contains numerical data. Based on the name of the column and values in it, I think the column is about the total amount of the commodity exported in crores. Similarly, column 'Exports in US millions' contains numerical values, and based on the name and values in it, I think it is about the amount of the commodity exported in US millions. Column 'Country' contains 'India' as the value, which is not a destination country, but rather the source country. There are no other columns in the table which have information about other countries. Thus, this table may not be directly useful for answering the question.\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'\n This table contains information about exports of select commodities from India to principal countries based on an old commodity classification. This table could be relevant to the question. Let's look into the table content. The 'Commodity name' column indicates the type of commodity being exported, such as 'Tobacco,' 'Carpets,' and 'Rice.' Based on the values in the column 'SourceYear', this column contains information about the year for which the data is available. Column 'Exports in usd' contains numerical values, and based on the name and values in it, I think it is about the amount of the commodity exported in US dollars. Similarly, the column 'Exports in rupees' contains numerical data. Based on the name of the column and values in it, I think the column is about the total amount of the commodity exported in Indian rupees. Column 'Country' contains 'India' as the value, which is not a destination country, but rather the source country. Column 'Export to country' contains values like 'U.S.A', 'Kuwait', etc. indicating different countries. So this column provides information about the destination country to which items are exported from India. The 'Year' and 'Yearcode' columns represent the financial year and the corresponding year of the export. Based on the above analysis, I think this table is relevant. Let's move to the next table.\n\n\n Step 2: Selecting relevant tables:\n\n Based on the analysis, it appears that the second table ('Trade and Balance of Payments - Exports of Select Commodities to Principal Countries (New commodity classification as per 2009-10) 7455') and the fourth table ('Trade and Balance of Payments - Exports of Select Commodities to Principal Countries (Old commodity classification) 7454') are relevant for determining India's export destination countries.\n\n\n Step 3: Determining relevant schema:\n\n Based on the question and information in these relevant tables, this is the relevant schema:\n 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'('Exports in rupees' , 'Commodity name' , 'Export of selected new item commodities to principal countries.' , 'Yearcode' )\n 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'('Commodity name' , 'Exports in usd' , 'Exports in rupees' , 'Export to country' , 'Yearcode' )\n\n\n Step 4: Finally, based on the question and the relevant schema, we can write the SQL as:\n\n WITH export_data AS ( SELECT 'Yearcode', 'Commodity name', 'Export of selected new item commodities to principal countries.' AS destination_country, SUM('Exports in rupees') AS total_exports FROM 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455' GROUP BY 'Year', 'Commodity name', destination_country UNION ALL SELECT 'Yearcode', 'Commodity name', 'Export to country' AS destination_country, SUM('Exports in rupees') AS total_exports FROM 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454' GROUP BY 'Year', 'Commodity name', destination_country ) SELECT 'Year', destination_country, sum(total_exports) FROM export_data GROUP BY 'Year', 'destination_country'"
    },
    {
        "role": "assistant",
        "content": "Thank you for providing the detailed solution. Based on the example, I understand the strategy to be used for writing the SQL query. Please go ahead and provide the next question along with the corresponding schema, and I'll assist you in writing the SQL query using the same strategy."
    },
    {
        "role": "user",
        "content": "Now, write SQL for the question mentioned below based on the corresponding schema information using the same strategy mentioned in the example previously.\n\n Schema information:\n Here are a few rows from each table in the schema. Keep in mind that the tables in the schema may have more rows than what is shown here.\n Table: 'India\’s Imports or Exports of Principal Agricultural Commodities 6165'\n 'Imports or exports' | 'Commodity type' | 'Value of imports or exports of principal agricultural commodities' | 'Quantity of imports or exports of principal agricultural commodities' | 'Country' | 'YearCode' | 'Year'\n 'Exports' | 'Oil meals' | '7831.79' | '4671.14' | 'India' | '2009' | 'Financial Year (Apr - Mar), 2009'\n 'Imports' | 'Fresh vegetables' | '11.12' | '8.55' | 'India' | '2016' | 'Financial Year (Apr - Mar), 2016'\n 'Imports' | 'Tobacco unmanufactured' | '69.47' | '1.54' | 'India' | '2017' | 'Financial Year (Apr - Mar), 2017'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'\n 'Commodity name' | 'Exports in usd' | 'Country' | 'Export to country' | 'Yearcode' | 'Year'\n 'Rice' | '31.0' | 'India' | 'U.S.A' | '2002' | 'Financial Year (Apr - Mar), 2002'\n 'Tea' | '75.3' | 'India' | 'Others' | '1996' | 'Financial Year (Apr - Mar), 1996'\n 'Iron Ore' | '0.0' | 'India' | 'Chile' | '2008' | 'Financial Year (Apr - Mar), 2008'\n\n Table: 'Trade and Balance of Payments - Imports of Principal Commodities ( New commodity classification as per 2009-10 ) 7453'\n 'Imports in usd' | 'Imports of principal commodities' | 'Imports in inr' | 'Country' | 'Yearcode' | 'Year'\n '14977.5' | 'Iron & Steel' | '97960' | 'India' | '2015' | 'Financial Year (Apr - Mar), 2015'\n '241.3' | 'Sulphur & Unroasted Iron Pyrts' | '1099' | 'India' | '2010' | 'Financial Year (Apr - Mar), 2010'\n '22732.5' | 'Transport equipment' | '146509' | 'India' | '2017' | 'Financial Year (Apr - Mar), 2017'\n\n Table: 'Trade and Balance of Payments - Imports of Principal Commodities ( Old commodity classification ) 7452'\n 'Imports of principal commodities' | 'Sourceyear' | 'Imports in rupees' | 'Country' | 'Year' | 'Imports in usd' | 'Yearcode'\n 'Crude Rubber, including Synthetic and Reclaimed' | '1991-92 ' | '182.0' | 'India' | 'Financial Year (Apr - Mar), 1991' | '73.8' | '1991'\n 'Organic and Inorganic Chemicals' | '2012-13 ' | '105141.0' | 'India' | 'Financial Year (Apr - Mar), 2012' | '19323.9' | '2012'\n 'Capital Goods' | '1988-89 ' | '6956.0' | 'India' | 'Financial Year (Apr - Mar), 1988' | '4803.0' | '1988'\n\n Table: 'Imports and Exports of Principal Agricultural Commodities of India 6129'\n 'Imports of agricultural commodities' | 'Percentage of agricultural imports to total national imports' | 'Agricultural exports to total national imports' | 'Country' | 'Yearcode' | 'Year'\n '54365.29' | '3.99' | '9.99' | 'India' | '2009' | 'Financial Year (Apr - Mar), 2009'\n '5890.1' | '4.8' | '19.18' | 'India' | '1995' | 'Financial Year (Apr - Mar), 1995'\n '6612.6' | '4.76' | '20.33' | 'India' | '1996' | 'Financial Year (Apr - Mar), 1996'\n\n Table: 'Selected Economic and Social Indicator 6159'\n 'Imports in US millions' | 'Country' | 'YearCode' | 'Year'\n '381008' | 'India' | '2015' | 'Financial Year (Apr - Mar), 2015'\n '50536' | 'India' | '2000' | 'Financial Year (Apr - Mar), 2000'\n '15869' | 'India' | '1980' | 'Financial Year (Apr - Mar), 1980'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'\n 'Commodity name' | 'Country' | 'Export of selected new item commodities to principal countries.' | 'Yearcode' | 'Year'\n 'Rice' | 'India' | 'France' | '2015' | 'Financial Year (Apr - Mar), 2015'\n 'Cashew' | 'India' | 'U.S.A' | '2013' | 'Financial Year (Apr - Mar), 2013'\n 'Cotton Yarn/Fabs./made-ups, Handloom Products etc.' | 'India' | 'Mauritius' | '2018' | 'Financial Year (Apr - Mar), 2018'\n\n Question: Which commodity had the highest imports in Old Commodity Classification?\n SQL Query:"
    }
]


# %%
for item in example_messages_2:
    print(item['role'])
    print(item['content'])
    print()

# %%
for model in [
    "gpt-3.5-turbo-16k-0613"
    ]:
    print(model)
    print(f"{num_tokens_from_messages(example_messages_2, model)} prompt tokens counted by num_tokens_from_messages().")

# %%
