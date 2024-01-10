import openai
openai.api_key = 'API - Key'
def get_chatgpt_responses(query, n_responses=5):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=query,
        max_tokens=150,
        n = n_responses
    )
    responses = [choice['text'].strip() for choice in response['choices']]
    
    return responses
query = input("Enter your query: ")

chatgpt_responses = get_chatgpt_responses(query, n_responses=1)

with open("Responses.txt", "w") as file:
    for i, response in enumerate(chatgpt_responses, start=1):
        file.write(f"Response {i}:\n{response}\n\n")

print("Responses written to chatgpt_responses.txt")
