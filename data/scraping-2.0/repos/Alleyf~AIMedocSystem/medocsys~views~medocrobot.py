import openai

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.gzip import gzip_page

openai.api_base = "https://openai.yugin.top/v1"


@gzip_page
@csrf_exempt
def chat(request):
    # 设置API密钥
    try:
        question = "你现在的身份是智能医学助理，" + request.POST.get(
            "question") + "并且检查你的回答是否存在语法错误，将不通顺的语法错误纠正后的内容发给我。回答结果中不用告诉我有无语法错误。"
        print(question)
        openai.api_key = "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"
        # 设置问题和上下文
        # 调用GPT-3生成答案
        response = openai.Completion.create(
            # engine="davinci",
            engine="text-davinci-003",
            prompt=question,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # 输出答案
        answer = response.choices[0].text.strip()
        context = {
            "status": 200,
            "answer": answer
        }
    except Exception as e:
        context = {
            'status': 403,
            'answer': "抱歉您当前的网络存在问题，请稍后重试，谢谢。"
        }
    # print(answer)
    return JsonResponse(context)
