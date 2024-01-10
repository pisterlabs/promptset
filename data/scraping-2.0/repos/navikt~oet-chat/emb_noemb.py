import openai, os, requests
import getpass

openai.__version__ # 0.28.0

openai.api_type = "azure"
# Azure OpenAI on your own data is only supported by the 2023-08-01-preview API version
openai.api_version = "2023-08-01-preview"

# Azure OpenAI setup
openai.api_base = "https://faggruppe-gpt.openai.azure.com/" # Add your endpoint here
openai.api_key = getpass.getpass() # Add your OpenAI API key here
deployment_id = "gpt-4" # Add your deployment ID here

# Azure Cognitive Search setup
search_endpoint = "https://sprakteknologi-ai-search.search.windows.net"; # Add your Azure Cognitive Search endpoint here
search_key = getpass.getpass(); # Add your Azure Cognitive Search admin key here


def setup_byod(deployment_id: str) -> None:
    """Sets up the OpenAI Python SDK to use your own data for the chat endpoint.

    :param deployment_id: The deployment ID for the model to use with your own data.

    To remove this configuration, simply set openai.requestssession to None.
    """

    class BringYourOwnDataAdapter(requests.adapters.HTTPAdapter):

        def send(self, request, **kwargs):
            request.url = f"{openai.api_base}/openai/deployments/{deployment_id}/extensions/chat/completions?api-version={openai.api_version}"
            return super().send(request, **kwargs)

    session = requests.Session()

    # Mount a custom adapter which will use the extensions endpoint for any call using the given `deployment_id`
    session.mount(
        prefix=f"{openai.api_base}/openai/deployments/{deployment_id}",
        adapter=BringYourOwnDataAdapter()
    )

    openai.requestssession = session

setup_byod(deployment_id)



def run_query(query, system_content, datasource):

    completion = openai.ChatCompletion.create(
        messages=[
                {"role": "system", "content":system_content},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": query},
                ],
        deployment_id=deployment_id,
        dataSources=[  # camelCase is intentional, as this is the format the API expects
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": search_endpoint,
                    "key": search_key,
                    "indexName": datasource,
                }
            }
        ]
    )

    return completion



query = "jeg skal arrangere et møte med varighet over 3 timer utenfor eget arbeidssted. får jeg dekket servering?"

system_content= '''Follow these instructions:
1) Answer question given from the user.
2) only give answers based on the context.
3) do not give answers based on your own knowledge.
4) stick to new norwegian.
'''

# datasource using embedding
emb = run_query(query, system_content, datasource="emb")

# datasource without embedding
noemb = run_query(query, system_content, datasource="noemb")

# svar
emb["choices"][0]['message']['content']
noemb["choices"][0]['message']['content']

# referanser
emb["choices"][0]['message']['context']['messages'][0]['content']
noemb["choices"][0]['message']['context']['messages'][0]['content']
