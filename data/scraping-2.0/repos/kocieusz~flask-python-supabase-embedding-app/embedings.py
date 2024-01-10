import os
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

load_dotenv()

# Initialize Supabase Client
url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')
supabase: Client = create_client(url, key)

# Initialize OpenAI Client
opeani_key = url = os.environ.get('OPENAI_KEY')
client = OpenAI(api_key=opeani_key)

# Get the text to concat:
# Get the selected columns from the database that will be concatenated to vectorize
response = supabase.table('raw_companies_database').select("name", "size", "industry", "id", "description").execute()
extracted_data = response.data
print(f'embeddings.py / extracted_data: {extracted_data}')

# Concatenate the values of the selected columns except the id that will be used to identify the company
def transform_dict(single_dict):
    company_id = single_dict['id']
    concatted_values = " ".join([str(value) for key, value in single_dict.items() if key != 'id'])

    # Return a dictionary with the id and the concatted values
    print(f'embeddings.py / transform_dict function: id:{company_id}, concatted_values:{concatted_values}')
    return {'id': company_id, 'concatted_for_embedding': concatted_values}

# Get the raw data from the supabase and transform it to a list of dictionaries with the id and the concatted values
# using the function above (transform_dict())
transformed_data = [transform_dict(d) for d in extracted_data]
print(f'embeddings.py / transformed_data: {transformed_data}')

# Iterate over the transformed data and create the embeddings
for element in transformed_data:
    inputed_text = element["concatted_for_embedding"]

    # Create the embeddings
    response = client.embeddings.create(
        input=inputed_text,
        model="text-embedding-ada-002"
    )

    # Get the embedding
    embeding = response.data[0].embedding
    # print(embeding)

    # Specify the company id, concatinated string and the embedding to insert in the database
    company_id = element["id"]
    input_text = element["concatted_for_embedding"]
    embeding_to = embeding

    print(f'embeddings.py / element loop: company_id:{company_id}, input_text:{input_text}, embeding_to:{embeding_to}')

    # Insert the embedding in the database
    response = supabase.table('embedings').insert({
        "company_id": company_id,
        "content": str(inputed_text),
        "embedding": embeding_to
    }).execute()