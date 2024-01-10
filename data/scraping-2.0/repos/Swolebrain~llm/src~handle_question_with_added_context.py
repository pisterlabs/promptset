import json
import os
from helpers import get_aws_auth, get_embeddings_with_retry
import requests
import openai

openai.api_key = os.environ.get('OPENAI_API_KEY')

def get_text_context_from_opensearch(prompt_embeddings, index_name, opensearch_endpoint_url, region, max_hits):
    auth=get_aws_auth(region)
    query = {
        "size": 4,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {
                        "field": "embeddings",
                        "query_value": prompt_embeddings,
                        "space_type": "cosinesimil"
                    }
                }
            }
        }
    }
    query_json = json.dumps(query)
    search_url = f'{opensearch_endpoint_url}/{index_name}/_search'
    headers = {'Content-Type': 'application/json'}

    # Send the POST request to the search endpoint with the query
    response = requests.post(search_url, headers=headers, data=query_json, auth=auth)
    if response.status_code != 200:
        print("Error getting similar embeddings: ", response.text)
        exit()

    response_json = json.loads(response.content)
    hits = response_json['hits']['hits']

    context_text = "/n".join([
        hit['_source'].get('text', '') for hit in hits[:max_hits]
    ])
    return context_text

def handle_question_with_added_context(question, index_name, opensearch_endpoint_url, region, max_context_hits = 3):
    prompt_embeddings = get_embeddings_with_retry(question)
    augmented_prompt_context = get_text_context_from_opensearch(prompt_embeddings, index_name, opensearch_endpoint_url, region, max_context_hits)
    gpt_response = openai.Completion.create(
        prompt="".join([
            """Answer the question based on the context below, and if the question can't be answered ba=sed on the context, say "I don't know"\n\nContext: """,
            augmented_prompt_context,
            "\n\n---\n\nQuestion: ",
            question,
            '\nAnswer: '
        ]),
        temperature=0.7,
        max_tokens=2500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        model='text-davinci-003',
    )
    print(json.dumps(gpt_response, indent=2))
    return gpt_response['choices'][0]['text'].strip()

