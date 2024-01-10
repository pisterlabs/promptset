from django.shortcuts import render, get_object_or_404, redirect
from ..models import Sum
import openai
import os
from dotenv import load_dotenv
from django.contrib.auth.decorators import login_required

# .envファイルを読み込む
load_dotenv()

@login_required
def index(request):
    return render(request, 'sumgpt/SpeechRecognition.html')

def sp(request):
    if request.method == 'POST':
        input_data = request.POST.get('data', '')

        # 要約2で文字起こししたデータはinput_dataに入っています
        # Google Docsと連携するときはこの下にコードを書くといい感じのはず

        # ChatGPTでの要約
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = "https://api.openai.iniad.org/api/v1"

        question = '次の文章を簡潔に要約して\n' + input_data

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": question},
            ],
        )
        speech_result = response['choices'][0]['message']['content']

        # dbに保存
        user = request.user
        sum_instance = Sum.objects.create(user=user, sum=speech_result, detail=input_data)

        # 保存したインスタンスのIDを取得
        saved_id = sum_instance.id
        sum_data = get_object_or_404(Sum, pk=saved_id)

        return render(request, 'sumgpt/sum.html', {'sum_data': sum_data})
    else:
        return render(request, 'sumgpt/sp.html')

def sum(request, pk):
    sum_data = get_object_or_404(Sum, pk=pk)  # SumモデルとデータのIDを指定
    return render(request, 'sumgpt/sum.html', {'sum_data': sum_data})

#お知らせの削除
@login_required
def sum_del(request, pk):
    Sum.objects.filter(id=pk).delete()
    return redirect('mypage')
