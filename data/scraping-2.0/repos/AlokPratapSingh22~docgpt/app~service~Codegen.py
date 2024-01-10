import json
import os

import tiktoken
import openai
import pandas as pd
import numpy as np
from dotenv import load_dotenv, find_dotenv
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from openapi_spec_validator import validate_spec
from jsonref import JsonRef

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")


def wrap_data(text: str) -> str:
    """
    Wrap string/json to reduce tokens
    """
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\\n', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('  ', ' ')
    return text


def parse_openapi_spec(spec_file: str) -> list[tuple[str, str]]:
    """
    Extract and split API details
    """

    # Load the Swagger JSON file
    with open(spec_file, "r") as file:
        spec = json.load(file)

    # Resolve all JSON references
    spec = JsonRef.replace_refs(spec)

    # TODO : Enable validation of specs before extracting info
    # Validate the spec
    # validate_spec(spec)

    openapi_version = spec.get("swagger", spec.get("openapi"))

    data = []

    # Parse all paths
    for path, path_content in spec.get('paths', {}).items():
        for method, method_content in path_content.items():
            if method in ['get', 'put', 'post', 'delete', 'options', 'head', 'patch', 'trace']:

                # Gather text info for embeddings
                operation_id = method_content.get('operationId', '')
                description = method_content.get('description', '')
                summary = method_content.get('summary', '')

                # Extract all type of parameters
                url_query_params = [p for p in path_content.get('parameters', []) if p['in'] == 'query']
                url_path_params = [p for p in path_content.get('parameters', []) if p['in'] == 'path']

                method_query_params = [p for p in method_content.get('parameters', []) if p['in'] == 'query']
                method_path_params = [p for p in method_content.get('parameters', []) if p['in'] == 'path']

                query_params = url_query_params + method_query_params
                path_params = url_path_params + method_path_params
                body_params = []

                if '3' in openapi_version:
                    request_body = method_content.get('requestBody', {})
                    if request_body:
                        content = request_body.get('content', {})
                        if 'application/json' in content:
                            body_params = content.get('application/json', {}).get('schema', {})
                else:
                    body_params = [p for p in method_content.get('parameters', []) if p['in'] == 'body']

                # Extract security block
                if 'security' in method_content:
                    security_keys = []
                    for key_dict in method_content['security']:
                        security_keys.extend(key_dict.keys())

                    if "3" in openapi_version:
                        security_spec = spec.get('components', {}).get('securitySchemes', {})
                    else:
                        security_spec = spec.get('securityDefinitions', {})

                    security = []
                    for key in security_keys:
                        if key in security_spec:
                            security.append({key: security_spec[key]})

                else:
                    security = None

                # Extract response structure
                responses = method_content.get('responses', {})

                txt = operation_id + ' ' + description + ' ' + summary
                details = str({
                        'operationId': operation_id,
                        'description': description,
                        'summary': summary,
                        'Endpoint': path,
                        'Request Method': method,
                        'Query parameters': query_params,
                        'Path parameters': path_params,
                        'Body parameters': body_params,
                        'Response structure': responses,
                        'Security': security
                    })

                data.append((wrap_data(txt), wrap_data(details)))

    return data


def tokenize(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Count number of tokens for each openapi doc block
    """
    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.doc.apply(lambda x: len(tokenizer.encode(x)))

    return df


def apply_embeddings(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Create embeddings for text column
    """
    df['embeddings'] = df.text.apply(
        lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

    return df


def process(filename: str) -> None:
    """
    Base method for preprocessing
    Steps:
        1. Extract all API details from uploaded OPENAPI JSON file
        2. Tokenize extracted paths
        3. Get Embeddings of text
        4. Write to CSV file
    """
    path_list = parse_openapi_spec(filename)

    # Create a dataframe from the list of texts
    df = pd.DataFrame(path_list, columns=['text', 'doc'])

    # Tokenize texts
    df = tokenize(df)

    # apply embeddings
    df = apply_embeddings(df)

    # Write to CSV file
    csv_filename = ''.join(filename.split('.')[:-1]) + '.csv'
    df.to_csv(csv_filename)


def collect_openapi_docs(
        question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["doc"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def query_codegen(
        filename,
        question="",
        model="text-davinci-003",
        max_len=2000,
        size="ada",
        debug=False,
        max_tokens=1500,
        stop_sequence=None

):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    df = pd.read_csv(filename, index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    context = collect_openapi_docs(
        question,
        df,
        max_len=max_len,
        size=size,
    )

    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        prompt = f"""Generate requested code based on openapi context below\n\nCONTEXT:\n{context}\n\n---\n\nQuestion: {question}\nAnswer:"""

        # print("prompt:\n" + prompt)
        # print("\n\n")
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=prompt,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""
