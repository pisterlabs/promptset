# test03-fix-grammar.py

import openai

# 각자 OPENAI API KEY 지정 : 이 파일은 버전 관리에는 절대 넣지 마세요.
openai.api_key = "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"

# API KEY 설정에 오류가 있는 지 확인하기 위함
print("api_key :", repr(openai.api_key))

# 텍스트 생성 혹은 문서 요약
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="""
Fix grammar errors:
- I is a boy
- You is a girl""".strip(),
)

print(response.choices[0].text.strip())