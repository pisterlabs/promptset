import openai

def generate_feedback(name, abilities):
    #openai.api_key = "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"
    
    short_name = name[:]  # 성 제거

    positive_feedback = ''
    gpt_positive_response = ''
    negative_feedback = ''
    gpt_negative_response = ''
    gpt_praise_encouragement_response = ''
    
    def get_gpt_response(prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "유치원생 대상으로 검사을 진행하는 선생님이고 유치원생 부모님에게 능력이 좋음부분 명확하게 설명하고 4줄 정도의 내용을 말해주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=500
        )
        return response.choices[0].message['content']

    positive_points = [ability for ability, level in abilities.items() if level == "좋음"]
    negative_points = [ability for ability, level in abilities.items() if level == "나쁨"]

    if positive_points:
        positive_text = "와 ".join(positive_points)
        positive_feedback = f"{short_name} 친구는 {positive_text}에서 뛰어난 모습을 보이고 있어요!"
        positive_response = positive_feedback + f" {short_name} 친구의 {positive_text} 능력은 어떤 활동에서 장점이 될 수 있는지 알려줘"
        gpt_positive_response = get_gpt_response(positive_response)

    if negative_points:
        negative_text = "와 ".join(negative_points)
        negative_feedback  = f"그러나 {short_name} 친구는 {negative_text}에서는 조금 부족한 모습을 보이고 있어요"
        negative_response = negative_feedback + f" {short_name} 친구의 {negative_text} 능력을 어떻게 보완할 수 있을지 알려줘"
        gpt_negative_response = get_gpt_response(negative_response)

    praise_encouragement = f"먼저 {name} 친구가 검사를 열심히 완료한 것에 칭찬을 해주세요!. 아직 부족한 부분이 있을 수 있지만, 그런 부분에 대해서 격려를 해주고 칭찬을 많이 해주세요!"
    gpt_praise_encouragement_response = get_gpt_response(praise_encouragement)

    text = positive_feedback + '\n' + gpt_positive_response + '\n' + negative_feedback + '\n' + gpt_negative_response + '\n' + gpt_praise_encouragement_response
    
    return text

### 호출 방식

# generate_feedback("child name", {
#     "주의력": "좋음",
#     "기억력": "보통",
#     "처리능력": "좋음",
#     "언어능력": "보통",
#     "유연성": "좋음"
# })
