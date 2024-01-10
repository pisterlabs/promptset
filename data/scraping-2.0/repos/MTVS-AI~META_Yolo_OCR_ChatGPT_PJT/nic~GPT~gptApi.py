import os
import openai
openai.api_key = "api_key"
# OCR 결과를 사용하여 분류 수행
ocr_results = []
ocr_text = "잠실역 도보 10분 거리 오피스텔 분양"

def classify_text(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"What category does this belong to: real estate, loans, entertainment?\n{text}"}
        ]
    )
    return response.choices[0].message['content']

# "The category is real estate."
print(classify_text(ocr_text))
# def main():
#   openai.api_key = "api-key"
#   response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#           {"role": "system", "content": "You are a helpful assistant."},
#           {"role": "user", "content": "Who won the world series in 2020?"},
#           {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#           {"role": "user", "content": "Where was it played?"}
#       ]
#   )

#   output_text = response["choices"][0]["message"]["content"]
#   print(output_text)