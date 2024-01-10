import os
import openai

def set_api_key(api_key):
    openai.api_key = api_key

def create_chat_messages(latex_formula):
    return [
        {"role": "system", "content": "당신은 복잡한 수학 문제를 이해하고 풀 수 있는 능력을 가진 지능형 조수입니다. LaTeX 표기법에 익숙합니다."\
          "그리고 수식의 각 부분을 분석하고, 그 의미를 설명할 수 있습니다. 또한, 수학적 원리를 기반으로 문제를 풀이하고, 이 과정을 단계별로 설명할 수 있습니다."},
        {"role": "assistant", "content": f"다음 LaTeX로 표현된 수식은 한국의 고등학교 수학 문제의 일부입니다."\
         " 이 수식을 해석하고 풀어줄 수 있나요? 그리고 해결 과정에 대한 자세한 설명도 부탁드립니다.: '{"+latex_formula+"}'"},
        #{"role": "assistant", "content": f"이 수식을 더 자세히 배우기 위해 어떤 웹사이트를 방문하는 것이 좋을까요? 수식에 대한 설명이 포함된 사이트를 추천 부탁드립니다."}
    
    ]

def generate_chat_completion(api_key, latex_formula, max_tokens=1000, temperature=0.2):
    set_api_key(api_key)
    messages = create_chat_messages(latex_formula)
    
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=max_tokens,  
        temperature=temperature  
    )

    return completion.choices[0].message["content"].strip()