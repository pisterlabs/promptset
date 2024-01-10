import os
from django.http import JsonResponse
from django.shortcuts import render
from openai import OpenAI
from dotenv import load_dotenv


def chat_with_gpt(request):
    user_input = request.GET.get('user_input')

    if not user_input:
        return JsonResponse({'response': 'Запит не може бути пистим, запитайте щось'})

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=1000
    )
    chat_response = chat_completion.choices[0].message.content
    return JsonResponse({'response': chat_response})


def consult(request):
    return render(request, 'consult.html')
