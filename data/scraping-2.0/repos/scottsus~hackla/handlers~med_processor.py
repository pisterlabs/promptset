import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPEN_AI_KEY')
)

def decomplicate(complex_medical_terms):
    payload = "Simplify this medical sentence and explain difficult medical jargons in parenthesis with format \"simplified: `sentence`\" and nothing else as your response: " + complex_medical_terms

    # Use GPT 4 to simplify medical terms
    res = client.chat.completions.create(
        model = "gpt-4",
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": payload}
        ]
    )

    simple_medical_terms = res.choices[0].message.content

    # # find the fist instance of 'simplified: ' and return the rest of the string excluding 'simplified: '
    simple_medical_terms = simple_medical_terms[simple_medical_terms.find('Simplified: ') + len('Simplified: '):]

    # print(simple_medical_terms)
    if simple_medical_terms == '':
        return (None, 'Unable to retrieve medical terms from server')

    return (simple_medical_terms, None)

# test
if __name__ == '__main__':
    print(decomplicate('With diaphragmatic involvement, splinting of the chest and pain in one or both shoulders may occur. Other manifestations of FMF include acute pleurisy (in 30 %); arthritis (in 25 %), usually involving the knee, ankle, and hip; an erysipelas - like rash of the lower leg; and scrotal swelling and pain caused by inflammation of the tunica vaginalis of the testis. Pericarditis occurs very rarely.'))