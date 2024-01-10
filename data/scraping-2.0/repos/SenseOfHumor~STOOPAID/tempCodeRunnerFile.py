import openai


api_key = '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' 


openai.api_key = api_key

def ask_question(question, context):
    response = openai.Completion.create(
        engine="text-davinci-002", 
        prompt=f"Question: {question}\nContext: {context}\nAnswer:",
        max_tokens=50, 
    )
    return response.choices[0].text.strip()


question = "What is the capital of France?"
context = "France is a beautiful country in Western Europe."
answer = ask_question(question, context)
print("Answer:", answer)
