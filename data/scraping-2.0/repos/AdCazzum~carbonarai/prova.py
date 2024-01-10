import requests
import openai

# Sostituisci '[api-key]' con la tua chiave API
api_key = ''

# URL del servizio GraphQL
url = f'https://api.thegraph.com/subgraphs/name/ensdomains/ens'

with open("backend/app/graphql/ens.root.object", "r") as f:
    q_roots = f.read()

with open("backend/app/graphql/ens.graphql", "r") as f:
    txt = f.read()

# print(txt)

# Funzione per generare la query GraphQL utilizzando GPT-3.5
def generate_graphql_query(p):
    prompt=f"""
    Given the following graphql schema:
    ```
    {txt}
    ```
        
    Translate the following into a syntactically valid graphql query.
    Try to not invent new fields, but use the ones already defined in the schema.
    Prefer less precise results over probably failing queries.
    Give me only the query source.
        
    ```
    ${p}
    ```
    """
    print(prompt)
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        # engine="davinci-002",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip().replace("`", "").strip()

retry = True
while retry:
        
    # Prompt per generare la query GraphQL
    prompt = "Give me the first 3 domains"
    query = generate_graphql_query(prompt)

    print("------------------")
    print(query)
    print("------------------")

    # Parametri della richiesta GraphQL
    variables = {}

    # Creazione della richiesta POST
    response = requests.post(url, json={'query': query, 'variables': variables})

    # Verifica della risposta
    if not response.status_code == 200:
        data = response.json()
        print(f'Errore nella richiesta GraphQL: {response.status_code}')
        print(response.text)
        retry = True  # Set retry to True to retry the request
    else:
        graphql_response = response.json()
        if "errors" in graphql_response:
            print("GraphQL response contains errors. Retrying...")
            retry = True
        else:
            retry = False  # Set retry to False to stop retrying

# Verifica della risposta
if not response.status_code == 200:
    data = response.json()
    print(f'Errore nella richiesta GraphQL: {response.status_code}')
    print(response.text)
    exit(1)


promptai = f"""
You are CarbonarAI, a friendly and helpful AI assistant by developed at EthRome2023 that provides help with interpreting GraphQL responses.
You give thorough answers. Use the following pieces of context to help answer the users question. If its not relevant to the question, provide friendly responses.
If you cannot answer the question or find relevant meaning in the context, tell the user to try re-phrasing the question. Use the settings below to configure this prompt and your answers.

<User Query>
{prompt}

<response>
```       
{response.text}
```
"""

print(promptai)

f = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=promptai,
        max_tokens=200
    )

print(f.choices[0].text)
