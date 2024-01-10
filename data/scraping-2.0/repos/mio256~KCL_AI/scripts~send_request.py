import os
import openai
import dotenv
import requests
import pprint

dotenv.load_dotenv()

openai.api_base = os.environ.get("AOAIEndpoint")

# Azure OpenAI on your own data is only supported by the 2023-08-01-preview API version
openai.api_version = "2023-08-01-preview"
openai.api_type = 'azure'
openai.api_key = os.environ.get("AOAIKey")


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


aoai_deployment_id = os.environ.get("AOAIDeploymentId")
setup_byod(aoai_deployment_id)

completion = openai.ChatCompletion.create(
    messages=[
        # {
        #     "role": "system",
        #     "content": "あなたは人々が情報を見つけるのを助けるAIアシスタントです。日本語の書類を取り出したら、日本語でよく読んで、日本語で答えてください。"
        # },
        {
            "role": "user",
            "content": input('>')
        }
    ],
    deployment_id=os.environ.get("AOAIDeploymentId"),
    dataSources=[  # camelCase is intentional, as this is the format the API expects
        {
            "type": "AzureCognitiveSearch",
            "parameters": {
                "endpoint": os.environ.get("SearchEndpoint"),
                "key": os.environ.get("SearchKey"),
                "indexName": os.environ.get("SearchIndex"),
            }
        }
    ]
)
pprint.pprint(completion)
print(completion.choices[-1].message.content)
print(completion.choices[-1].message.context.messages[-1].content)
