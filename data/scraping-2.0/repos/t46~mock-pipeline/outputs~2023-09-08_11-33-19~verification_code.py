import openai
import numpy as np

questions = [
    ("What is the weather like?", "What is the current temperature in New York?"),
    ("Tell me about dogs.", "What is the average lifespan of a Labrador Retriever?"),
    ("What's happening in the world?", "What are the current top news headlines?"),
]

general_responses = []
specific_responses = []

for general, specific in questions:
    general_response = openai.Completion.create(engine="text-davinci-002", prompt=general, max_tokens=100)
    general_responses.append(general_response.choices[0].text.strip())

    specific_response = openai.Completion.create(engine="text-davinci-002", prompt=specific, max_tokens=100)
    specific_responses.append(specific_response.choices[0].text.strip())

for i in range(len(questions)):
    print(f"General question: {questions[i][0]}")
    print(f"General response: {general_responses[i]}")
    print(f"Specific question: {questions[i][1]}")
    print(f"Specific response: {specific_responses[i]}")
    print("\n")

def calculate_variance(responses):
    lengths = [len(response) for response in responses]
    return np.var(lengths)

general_variance = calculate_variance(general_responses)
specific_variance = calculate_variance(specific_responses)

print(f"Variance of general responses: {general_variance}")
print(f"Variance of specific responses: {specific_variance}")

if specific_variance < general_variance:
    print("The results support the hypothesis that making a question prompt more specific can lead to more precise responses from the LLM.")
else:
    print("The results do not support the hypothesis that the specificity of the question prompt significantly influences the precision of the LLM's responses.")