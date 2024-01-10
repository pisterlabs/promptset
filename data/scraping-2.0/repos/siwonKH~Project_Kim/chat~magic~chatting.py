import openai

chatting_prompts = [
    {
        'role': 'system',
        'content': '''너는 취업 상담사야.

사용자가 원하는 직무를 물어봐.
직무를 알았다면 사용자가 원하는 근무형태를 물어봐. 상세하게 물어보지는 않아도 돼.
근무형태를 알았다면 사용자의 학력을 물어봐.
학력을 알았다면 취업을 원하는 지역을 물어봐.


모두 알게 되었다면 계속 대화를 이끌어 나가며 적극적으로 상대방에게 질문해야 해.'''},
    {'role': 'assistant', 'content': "안녕하세요. 저는 당신에게 알맞은 취업 정보를 제공해 드릴 김비서 입니다."}
]


def stream_chatting_response(messages: list):
    messages[:0] = chatting_prompts
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=512,
        temperature=0.4,
        top_p=1,
        frequency_penalty=0.9,
        presence_penalty=0.4,
        messages=messages,
        stream=True,
    )
    for chunk in completion:
        if "content" in chunk['choices'][0]['delta']:
            content = str(chunk['choices'][0]['delta']['content'])
            yield f"{content}"


def chatting_response(messages: list) -> str:
    messages[:0] = chatting_prompts
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=512,
        temperature=0.4,
        top_p=1,
        frequency_penalty=0.9,
        presence_penalty=0.4,
        messages=messages
    )
    return completion['choices'][0]['message']['content']
