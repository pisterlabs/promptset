import os
import json
import openai

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "null")
API_URL = "https://api.openai.com/v1/chat/completions"
# Define the model in the client
# model = "gpt-3.5-turbo"
# model = "gpt-4-0613"

openai.api_key = os.getenv("OPENAI_API_KEY")

init_system_prompt = """
    You are an expert software engineer who is explaining and sometimes writing documentation that describes a code repository.
    You will be provided with some code snippets or documents from the repository.  Answer the user's questions to the best of your ability.
    You can ask the user for clarifying information if it is unclear what they want.
    You should modify your response based on the user's level of experience.  You can ask the user for their level of experience with software.
    You should always try to provide examples from the code or documents provided in order to help support your answer.
    When you quote the snippets always provide the file name.
"""

def query_llm(query, context_docs, msgs, model):
    print(f"Sending request to OpenAI API... {model}")
    prompt = f"Here is the user's question that you should attempt to answer: {query}\n\n"
    prompt += f"And here are a few code and text snippets from the repository that are relevant to the question.  These snippets are not sorted in any particular order: \n\n"
    for idx, doc in enumerate(context_docs):
        prompt += f"Code snippet #{idx+1}:\n"
        prompt += f"  Filename: {doc.metadata['source']}\n"
        prompt += f"  Contents: {doc.page_content}\n"
        prompt += "\n\n"
    token_limit = 8000
    if len(prompt) > token_limit:
        print("Prompt too long.  Truncating...")
        prompt = prompt[:token_limit]
#         print(prompt)
#     print("\n\nPrompt: ", prompt)

    messages = [{"role": 'system', "content": init_system_prompt}]
    for msg in msgs[:10]:
#         print(json.dumps(msg))
        messages.append({"role": msg['role'].lower(), "content": msg['msg']})
#     print(json.dumps(messages))

    # Add the user's current query to then end of messages
    messages.append({"role": "user", "content": f"{prompt}"})
#     print("\nmsgs from client:")
#     for msg in msgs:
#         print(json.dumps(msg))
#
#     print("\nmessages sent to OAI:")
#     for msg in messages:
# #         print(json.dumps(msg))
#         print( msg['role'] + ":" + msg['content'][0:20])
#     print('\n')

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=1,
        top_p=0.5,
    )
    print("Response received from OpenAI API.")
    assistant_response = response.choices[0].message.content
    return {
        'assistant_response': assistant_response,
        'usage': response.usage
    }
