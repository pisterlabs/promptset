import openai
import os
import sys
import json

def make_openai_request(data):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"faça uma petição de {data['petitionType']} para {data['clientName']}, processo numero {data['processNumber']}, na comarca {data['courtDistrict']}, advogado(a) {data['lawyerName']} numero OAB {data['oabNumber']}, informações gerais do processo:\n{data['caseFacts']}\n"
            }
        ],
        temperature=0.8,
        max_tokens=7022,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data = json.loads(sys.argv[1])
        result = make_openai_request(data)
        print(result)
    else:
        print("Nenhum dado fornecido.")
