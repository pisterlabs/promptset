import openai  # for calling the OpenAI API

GPT_MODEL = "gpt-4-1106-preview"

text = open('esi-implementation-handbook-2020.txt', 'r').read()

question = """
What data is needed to triage a patient using ESI?
"""

query = f"""Use the ESI Implementation handbook below as a reference text. If the answer cannot be found, write "I don't see those details in text, but I think..." and try to make your best guess as to what the right answer would be.

Article:
\"\"\"
{text}
\"\"\"

Question: {question}
"""

response = openai.chat.completions.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about ESI triage implementation related to software implementation.'},
        {'role': 'user', 'content': query},
    ],
    model=GPT_MODEL,
    temperature=0,
)

print(response.choices[0].message.content)

