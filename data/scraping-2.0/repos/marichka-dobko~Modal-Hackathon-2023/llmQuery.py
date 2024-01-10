from claude_api import Client
from mongoQuery import get_result
import openai


# client = anthropic.Client(claude_key)


def get_llm_result(query):
    print("Start get llm result")
    
    res = get_result(query)
    user_query = query
    context1 = res[0]['Context']
    fileName1 = res[0]['Filename'].split('/')[-1]
    context2 = res[1]['Context']
    fileName2 = res[1]['Filename'].split('/')[-1]

    summary_prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"User has the following question: {user_query}. Answer the question with these 2 responses. First context: {context1} from Policy {fileName1}, second context: {context2} from Policy {fileName2}. Make a clean, cohesive, concise, beautify response and include the policy name."}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=summary_prompt
    )
    
    response_res = response['choices'][0]['message']['content']
    
    print("LLM query successful")
    return response_res

# def get_llm_result(query):
#     print("Start get llm result")
#     res = get_result(query)
#     user_query = query
#     context1 = res[0]['Context']
#     fileName1 = res[0]['Filename']
#     fileName1 = fileName1.split('/')[-1]
#     context2 = res[1]['Context']
#     fileName2 = res[1]['Filename']
#     fileName2 = fileName2.split('/')[-1]

#     summary_prompt = f"{anthropic.HUMAN_PROMPT} User has the following question {user_query} Answer the question with these 2 responds first context {context1} from Policy {fileName1},second context {context2} from Policy {fileName2} and make a clean cohesive, concise, beautify respond and include the policy name{anthropic.AI_PROMPT}"

#     response = client.completion(
#         prompt=summary_prompt,
#         stop_sequences=[anthropic.HUMAN_PROMPT],
#         model="claude-v1-100k",
#         max_tokens_to_sample=5000,
#     )
#     response_res = response['completion']

#     print("LLM query successful")
#     return response_res