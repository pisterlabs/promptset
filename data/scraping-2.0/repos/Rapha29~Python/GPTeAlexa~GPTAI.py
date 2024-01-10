
# 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

import tika
from tika import parser
import openai

openai.api_key = "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"

tika.initVM()
parsed_pdf = parser.from_file('livro.pdf')
text = parsed_pdf['content']

def answer_question(question, context, model="text-davinci-002", max_len=2048, debug=False):
    try:
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below:\n\n{context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0.7,
            max_tokens=max_len,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model=model,
        )

        answer = response.choices[0].text.strip()
    except Exception as e:
        if debug:
            print(e)
        answer = "Não consegui encontrar a resposta no PDF."
    return answer

question = input("Qual pergunta você tem sobre o texto?\n")
answer = answer_question(question, text)
print(answer)