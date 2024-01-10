import openai
from django.http import JsonResponse, HttpResponse
from django.shortcuts import get_object_or_404
from rest_framework.response import Response
from rest_framework.decorators import api_view

from ..models import Problem, User, Rival,Solved
from ..serializers.problems import UnSolvedMoreSerializers,RecProblemPageSerializers,SimpleProblemList,RecProblemSerializers,SolvedProblemSerializers,UnSolvedSerializers
import time
import re
@api_view(['POST'])
def hint(request):
    openai.api_key = "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"
    problem_id = request.data["problem_id"]


    prompt = f"""나는 지금 알고리즘 문제를 풀고 있어,
    내가 풀고 있는 문제의 링크는 다음과 같아
    'https://www.acmicpc.net/problem/{problem_id}'
    하지만 나는 바로 정답을 알고 싶지 않고 힌트를 3번에 걸쳐서 줬으면 좋겠어, 힌트 형식은 3차례에 걸쳐서 줘 
    '힌트만 주면 돼'
    
    예시를 들어서
    'https://www.acmicpc.net/problem/2667'의 문제의 경우 다음과 같이 대답을 해주면 돼
    '
    1. 해당 문제는 BFS 또는 DFS 알고리즘을 사용하는 문제입니다.
    2. 방문 여부를 체크 해주는 변수를 같이 사용하면 더 쉽게 풀 수 있습니다.
    3. 주어진 맵의 범위를 벗어 나기 쉽기 때문에 이점을 조심해야합니다.
    '
    '알겠다는 대답없이 힌트만 개행으로 구분해서 알려줘.
    힌트 앞에는 1. 2. 3. 이렇게 숫자를 꼭 적어주고 형식은 꼭 지켜줘'
    """


    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    result = completion.choices[0].message.content

    hint1, hint2, hint3 = result.split("\n")[:3]
    return JsonResponse({"hint1":hint1,"hint2":hint2,"hint3":hint3})

    
@api_view(['GET'])
def list_rec(request):
    cur_user = request.user
    serializer = RecProblemPageSerializers(cur_user)
    time.sleep(1)
    return JsonResponse({"user":cur_user.username,**serializer.data})


@api_view(['GET'])
def list_rec_more(request):
    cur_user = request.user
    serializer = RecProblemSerializers(cur_user)

    return JsonResponse({"user": cur_user.username, **serializer.data})


@api_view(['GET'])
def list_problem(request):
    problems = Problem.objects.all()
    serializers = SimpleProblemList(problems, many=True)

    return Response(serializers.data)

@api_view(['GET'])
def list_unsolved(request):
    cur_user = request.user
    serializer = UnSolvedSerializers(cur_user)

    return JsonResponse({"user":cur_user.username,**serializer.data})


@api_view(['GET'])
def list_unsolved_more(request):
    cur_user = request.user
    serializer = UnSolvedMoreSerializers(cur_user)

    return JsonResponse({"user":cur_user.username,**serializer.data})




