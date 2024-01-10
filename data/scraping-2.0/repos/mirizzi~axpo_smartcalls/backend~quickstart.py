#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://smartcallopenai.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "d93fa35ed1ee488fb9a27a0704ef0c6e"

response = openai.ChatCompletion.create(
    engine="SmartCallsGPT4-test",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
        {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
        {"role": "user", "content": "Do other Azure Cognitive Services support this too?"}
    ]
)

print(response)
print(response['choices'][0]['message']['content'])