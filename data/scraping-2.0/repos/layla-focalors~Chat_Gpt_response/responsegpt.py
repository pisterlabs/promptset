import openai
openai.api_key = ""
def responser(content:str):   
    content = "너는 이아라는 버튜얼 유튜버라고 생각하고 답변해줘. 이아의 성격은 부드럽고 세상일에 무관심한 20세 대학생이야, 이아는 노래를 잘해 다음의 답변에 이아일때, 답해 봐 "+ content
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages = [{"role":"user", "content": f"{content}"}],
        # max_tokens = 20
    )
    return response.choices[0].message.content
# print(responser("시청자에 대한 인사를 해줘!"))
    
# print(responser("아름다운 노래를 읊어줘!"))