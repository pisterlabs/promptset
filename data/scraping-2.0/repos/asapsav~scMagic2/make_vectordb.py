import pandas as pd
import chromadb
import requests
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import tiktoken
import os
import dotenv
dotenv.load_dotenv()


def num_tokens_from_string(string: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Define a function to fetch README content
def get_readme(url):
    if pd.isna(url) or url == "":
        return ""
    try:
        # Construct URL to the raw README file
        readme_url = url.replace('github.com', 'raw.githubusercontent.com').rstrip('/') + '/master/README.md'
        response = requests.get(readme_url)
        if response.status_code == 200:
            return response.text
        else:
            return ""
    except Exception:
        # Handle exceptions
        return ""

def save_dataframe(df, directory, base_filename):
    """
    Save a DataFrame to a CSV file with versioning if the file already exists.

    :param df: DataFrame to be saved.
    :param directory: Directory where the file will be saved.
    :param base_filename: The base name of the file, without version number.
    """
    # Construct the initial file path
    file_path = os.path.join(directory, base_filename)

    # Initialize version number
    version = 0

    # Check if the file exists and update the file name with the next version
    while os.path.exists(file_path):
        version += 1
        # Construct new file path with version number
        file_name, file_extension = os.path.splitext(base_filename)
        file_path = os.path.join(directory, f"{file_name}_v{version}{file_extension}")

    # Save the DataFrame
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved as {file_path}")

# Prepare the DataFrame

def make_desc(row):
    # Example function using row 'A' and 'B'
    return 'Platform: ' + row['Platform'] + \
            '\n Description: ' + row['Description'] + \
            '\n Categories: ' + row['Categories']


def make_desc_readme(row):
    # Example function using row 'A' and 'B'
    return 'Platform: ' + row['Platform'] + \
            '\n Description: ' + row['Description'] + \
            '\n Categories: ' + row['Categories'] + \
            '\n Readme: ' + row['Readme']

def prepare_dataframe(df):

    counter = 0 # for status printing
    for index, row in df.iterrows():
        df.at[index, 'Readme'] = get_readme(row['Code'])

        # Increment the counter
        counter += 1
        if counter % 100 == 0:
            print(f"Processed {counter} rows.")
        
    df['extented_desc'] = df.apply(make_desc, axis=1)
    df['extented_desc_readme'] = df.apply(make_desc_readme, axis=1)

    assert df.isna().sum()['extented_desc'] == 0 # check for nans in desc to make embedding collection

    df['tokens_in_ext_desc'] = df['extented_desc'].apply(num_tokens_from_string)

    # save data frame with full readmes
    #save_dataframe(df, 'dataframes', 'tool-table-with-readmes.csv') 

    #df['tokens_in_ext_desc'].hist() # for jupyer notebook

    # Trim ReadMe to fit 8k tokens limit in embedding db - find better embedding model later
    df['extented_desc_readme_trim'] = df['extented_desc_readme'].apply(lambda x: x[:22000] if pd.notna(x) else x)

    df['tokens_in_ext_desc_readme_trim'] = df['extented_desc_readme_trim'].apply(num_tokens_from_string)

    #df['tokens_in_ext_desc_readme_trim'].hist(bins = 50)# for jupyer notebook

    assert df['tokens_in_ext_desc'].sum() < 100000
    # as if Nov 7 2023 allshort  descriptions of tools is jsut 68K tokens, so
    # no need for embed db truly speaking, can also just make claude calls every time need to pick a tool

    assert df.Name.nunique() == df.shape[0]

    # for jupyer notebook
    # df['extented_desc'].isna().sum() 

    # df['Code'].isna().sum()

    # df['Readme'].isna().sum()

    # df['Readme'].head(20)

    df['Readme'] = df['Readme'].apply(lambda x: x.replace('\n', ' ').replace('\r', ' ').replace("'", "\\'") if pd.notna(x) else x)

    # save data frame with trimmed readmes
    #save_dataframe(df, 'dataframes', 'tool-table-with-readmestrimmed.csv')
    return df

# Create Vector DB
def get_vector_db(df):

    # Uncomment for persistent client
    chroma_client = chromadb.PersistentClient()

    EMBEDDING_MODEL = "text-embedding-ada-002"
    # change this to biotech specialised model later
    embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name=EMBEDDING_MODEL)
    scrnatools_description_collection = chroma_client.create_collection(name='scRNA_Tools', embedding_function=embedding_function)

    # Add the content vectors
    scrnatools_description_collection.add(
        documents = list(df['extented_desc']),
        metadatas = df.drop(['extented_desc'], axis = 1).to_dict(orient='records'),
        ids = list(df.Name)
    )

    scrnatools_description_collection.add(
        documents = list(df['extented_desc_readme_trim']),
        metadatas = df.drop(['extented_desc_readme_trim'], axis = 1).to_dict(orient='records'),
        ids = list(df.Name))
    
    return scrnatools_description_collection

# Query DB
def query_collection(collection, query, max_results, dataframe):
    results = collection.query(query_texts=query, n_results=max_results, include=['distances'])
    result = pd.DataFrame({
                'id':results['ids'][0],
                'score':results['distances'][0],
                'content': dataframe[dataframe.Name.isin(results['ids'][0])]['extented_desc'],
                'platform': dataframe[dataframe.Name.isin(results['ids'][0])]['Platform'],
                })

    return result


# Import table with tools from scrna-tools.org
# you can download it from https://scrna-tools.org/tables/ (don't forget to select all columns)
# or access souce programmatically from https://github.com/scRNA-tools/scRNA-tools/tree/master/database
df = pd.read_csv('tableExport.csv')

try:
    df_enriched = prepare_dataframe(df)
    save_dataframe(df_enriched, 'dataframes', 'tool-table-with-readmestrimmed.csv')
except Exception as e:
    print(e)

try:
    scrnatools_description_collection = get_vector_db(df_enriched)
except Exception as e:
    print(e)


print(query_collection(scrnatools_description_collection, 'quality controll, python, 10M cells, human cells', 5, df))