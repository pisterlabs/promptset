from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextCompletion,
)
import semantic_kernel as sk

from utils.constants import OPENAI_DEPLOYMENT_NAME, OPENAI_ENDPOINT, OPENAI_API_KEY
from semantic_kernel.orchestration.context_variables import ContextVariables

def print_ai_services(kernel):
    print(f"Text completion services: {kernel.all_text_completion_services()}")
    print(f"Chat completion services: {kernel.all_chat_services()}")
    print(
        f"Text embedding generation services: {kernel.all_text_embedding_generation_services()}"
    )

kernel = sk.Kernel()

# Not availble for gtp-35-turbo-16k:  More info: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#gpt-35-models
# kernel.add_text_completion_service(
#     service_id="azure_gpt35_text_completion",
#     service=AzureTextCompletion(
#         OPENAI_DEPLOYMENT_NAME, OPENAI_ENDPOINT, OPENAI_API_KEY
#     ),
# )

gpt35_chat_service = AzureChatCompletion(
    deployment_name=OPENAI_DEPLOYMENT_NAME,
    endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_API_KEY,
)

kernel.add_chat_service("azure_gpt35_chat_completion", gpt35_chat_service)

print_ai_services(kernel)


prompt = """
Give me a good use case to learn {{$input}}.

I will use it to learn {{$input}}.

I am using python.

Provide the use case in bullet points, each bullet point corresponding to a semantic or native function.
"""

generate_capital_city_text = kernel.create_semantic_function(
    prompt, max_tokens=4096, temperature=0, top_p=0
)

response = generate_capital_city_text("semantic kernel")

if response.error_occurred:
    print(response.last_error_description)
else:
    print(response.last_error_description)

print(response)

print("end")