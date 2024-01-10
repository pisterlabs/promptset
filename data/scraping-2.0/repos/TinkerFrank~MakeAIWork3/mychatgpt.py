
# response = openai.ChatCompletion.create(
#     engine="MyChatGPT35Turbo", # engine = "deployment_name".
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
#         {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
#         {"role": "user", "content": "Do other Azure Cognitive Services support this too?"}
#     ]
# )


# print(response)
# print(response['choices'][0]['message']['content'])

import tiktoken
import openai
import os
openai.api_type = "azure"
openai.api_version = "2023-05-15"
# Your Azure OpenAI resource's endpoint value .
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

system_message = {"role": "system", "content": "You are a helpful assistant."}
max_response_tokens = 250
token_limit = 4096
conversation = []
conversation.append(system_message)
table_input = {"role": "system", "content": "{'Normal Apple': 0.999316930770874, 'Scab Apple': 0.0005993539816699922, 'Bad Apple': 5.529150803340599e-05, 'Rot Apple': 2.8484791982918978e-05}"}
conversation.append(table_input)


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        # every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


while (True):
    print('Input:')

    user_input = input("")

    # write exit statement
    if (user_input == 'q'):
        break

    conversation.append({"role": "user", "content": user_input})
    conv_history_tokens = num_tokens_from_messages(conversation)

    while (conv_history_tokens+max_response_tokens >= token_limit):
        del conversation[1]
        conv_history_tokens = num_tokens_from_messages(conversation)

    response = openai.ChatCompletion.create(
        # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
        engine="MyChatGPT35Turbo",
        messages=conversation,
        temperature=.7,
        max_tokens=max_response_tokens,
    )

    conversation.append(
        {"role": "assistant", "content": response['choices'][0]['message']['content']})
    print("\n" + response['choices'][0]['message']['content'] + "\n")
