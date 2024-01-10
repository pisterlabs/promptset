import openai
openai.api_key = "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"

keyword_string = "정말 맛있어요, 가성비가 좋아요, 친구와 함께, 점심식사로"
char_num = "200"
restaurant_name = "아건"

response = openai.ChatCompletion.create(
    model="gpt-4-0314",
    temperature=0,
    max_tokens=2048,
    messages=[
        {"role": "system", "content": "You are a helpful assistant in writing automated reviews for restaurants, "
                                      "with given keywords and instructions."},
        {"role": "user", "content": "Create me a restaurant review for '%s' restaurant, "
                                    "that includes '%s' keywords in korean. "
                                    "Do not include additional information that are not mentioned."
                                    "The restaurant review must be minimum %s characters(in korean)."
                                    % (restaurant_name, keyword_string, char_num)}
    ],
)

print(response["choices"][0]["message"]["content"])