import openai
import json


# 요청 내용 - 백준 문제 | https://www.acmicpc.net/problem/2798
text_string = """
Python 3
n “Blackjack”, a popular card game, the goal is to have cards which sum up to largest number not exceeding 21. Mirko came up with his own version of this game.

In Mirko‟s game, cards have positive integers written on them. The player is given a set of cards and an integer M. He must choose three cards from this set so that their sum comes as close as possible to M without exceeding it. This is not always easy since there can be a hundred of cards in the given set.

Help Mirko by writing a program that finds the best possible outcome of given game.


The first line of input contains an integer N (3 ≤ N ≤ 100), the number of cards, and M (10 ≤ M ≤ 300 000), the number that we must not exceed.

The following line contains numbers written on Mirko‟s cards: N distinct space-separated positive integers less than 100,000.

There will always exist some three cards whose sum is not greater than M.


The first and only line of output should contain the largest possible sum we can obtain.
"""
# 문제 끝

# main code
"""
API request & response

engine: 사용할 엔진, GPT-3의 자식이라고 할 수 있는 Codex 엔진 사용 / code-davinci-002가 가장 좋은 엔진이지만 신청 후 허가 필요
 따라서 지금은 code-davinci-edit-001 사용
temperature: 값이 낮을수록 안정적인 값 출력
instruction: 요청 내용
"""
response = openai.Edit.create(engine="code-davinci-edit-001", temperature=0, instruction=text_string)

# json 파싱
json_object = json.loads(response.__str__())
result_text = json_object['choices'][0]['text']

#결과 출력
print(result_text)