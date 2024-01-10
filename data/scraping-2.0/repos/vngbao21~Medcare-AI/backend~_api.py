from flask import jsonify
import openai

# get keys from .env file
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

model_id = "ft:gpt-3.5-turbo-0613:personal::8VMZbplW" 

def get_from_openAI(data): 
    try:
        completion = openai.ChatCompletion.create(
            model=model_id,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Bạn là một con chatbot bác sĩ thân thiện chuyên chẩn đoán sức khỏe dựa trên mô tả của bệnh nhân. Nếu bạn không biết điều gì, chỉ cần nói, Tôi không thể đưa ra chẩn đoán dựa trên thông tin bạn cung cấp."},
                {"role": "user", "content": data['prompt']},
            ]
        )

        data['response'] = completion.choices[0].message["content"]
        return jsonify(data)

    except Exception as e:
        data['response'] = f"Error: {e}"
        return jsonify(data)