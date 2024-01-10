import openai

running_cost = 0
running_prompt_tokens = 0
running_completion_tokens = 0

model4="gpt-4-1106-preview"
model3="gpt-3.5-turbo-1106"
current_model = model4
COSTS = {
    "gpt-3.5-turbo-1106": {"input": 0.001 / 1000, "output": 0.002 / 1000},
    "gpt-4-1106-preview": {"input": 0.01 / 1000, "output": 0.03 / 1000},
}

def llm_call(messages):
    global running_cost

    response = openai.ChatCompletion.create(
        model=current_model,
        messages=messages
    )

    calc_cost(response,current_model)
    
    return response['choices'][0]['message']['content']

def llm_call_with_functions(message_array, function_list):
        response = openai.ChatCompletion.create(
            model=current_model,
            messages=message_array,
            functions=function_list,
            function_call="auto"
        )

        calc_cost(response, current_model)

      
        
        return response["choices"][0]["message"]

def calc_cost(response, current_model):
    prompt_tokens = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']

    global running_prompt_tokens
    global running_completion_tokens
    running_prompt_tokens += prompt_tokens
    running_completion_tokens += completion_tokens

        #calculate the cost
        #prompt tokens cost 0.01 per 1000
        #completion tokens cost 0.003 per 1000
    
    prompt_cost = prompt_tokens * COSTS[current_model]["input"]
    completion_cost = completion_tokens * COSTS[current_model]["output"]
    total_cost = prompt_cost + completion_cost


    #print the total cost in dollars
    print("Query cost: $", total_cost)
    print("Prompt tokens: ", prompt_tokens)
    print("Completion tokens: ", completion_tokens)
    global running_cost
    running_cost += total_cost
    print("Total cost: $", running_cost)
    print("Total prompt tokens: ", running_prompt_tokens)
    print("Total completion tokens: ", running_completion_tokens)