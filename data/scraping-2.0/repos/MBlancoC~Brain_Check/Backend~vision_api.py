import os
from openai import OpenAI

def analyze_image_with_gpt4(image_data, question):
    client = OpenAI(api_key="tu_clave_api_aquí")

    # Codificar la pregunta a UTF-8
    utf8_question = question.encode('utf-8').decode('utf-8')

    response = client.completions.create(
        model="gpt-4-vision-preview",
        prompt=[
            {
                "role": "user",
                "content": utf8_question
            },
            {
                "role": "system",
                "content": {"type": "image", "data": image_data}
            }
        ],
        max_tokens=300
    )
    return response
