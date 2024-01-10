from openai import OpenAI

'''
openAI input/output format: {'role': '[ROLE]', 'content': '[CONTENT]'} 
    where [ROLE] and [CONTENT] are filled in.
'''

def clean_input(system_prompt, prompts):
    '''
    converts unformatted prompts into formatted ones ready to be passed to the LLM
    '''
    conversation = []
    inputs = [system_prompt]
    expected = [[]] # start with empty for system prompt
    conversation.append({'role': 'system', 'content': system_prompt})
    for prompt in prompts:
        inputs.append(prompt['prompt'])
        conversation.append({'role': 'user', 'content': prompt['prompt']})
        expected.append(prompt['expected'])
    return inputs, conversation, expected

def run_model(env_var, messages, temp_bench):
    '''
    calls the API and returns the prompt/response
    '''
    conversation = []
    
    if temp_bench == 1:
        temp = 1
    else:
        temp = float(env_var['TEMPERATURE'])
        
    client = OpenAI(api_key = env_var['API-KEY'])
    for message in messages:
        conversation.append(message)
        chat_completion = client.chat.completions.create(
            model = env_var['MODEL'],
            messages=conversation,
        )
        conversation.append({'role': 'assistant', 'content': chat_completion.choices[0].message.content})
    return conversation

def clean_output(response):
    '''
    converts the LLM-formatted prompts into strings of content only
    '''
    responses_only = []
    for comment in response:
        if comment['role'] == 'assistant':
            responses_only.append(comment['content'])
    return responses_only