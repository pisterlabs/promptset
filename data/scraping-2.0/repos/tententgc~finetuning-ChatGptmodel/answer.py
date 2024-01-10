import openai
openai.api_key = "YOUR API KEYS"
model_name = "davinci:ft-personal-2023-05-02-03-05-55"


def answer_question(prompt):
    response = openai.Completion.create(
        engine=model_name,
        prompt=prompt,
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

question = "CMSK อยู่ที่ไหน"
answer = answer_question(question)
print(f"Answer: {answer} +:::::::")
