import openai

from simple_detail import GetSimpleDetail

def gpt(delivery, size_options, best_review, template):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.9,
        messages=[{"role": "system", "content": template},
                  {"role": "user", "content": f"{delivery} {size_options} {best_review}"}],
        max_tokens=200  # 원하는 최대 토큰 수를 설정
    )

    return response.choices[0].message.content.strip()

def SimpleDetail(apikey, id):
    openai.api_key = apikey
    url = 'https://www.musinsa.com/app/goods/' + id
    simple_detail = GetSimpleDetail(url)
    delivery = simple_detail["delivery"]
    size_options = str(simple_detail["size_info"])
    best_review = str(simple_detail["best_review"])
    
    input = delivery + size_options + best_review
    print(input)

    simple_detail_template = """You are the helpful agent that summarizes product information from input contents. You should show size options. Don't add specific length(cm) for size. You should let customers know the date of delivery arrival. You should show the summary of the review if review is not None.
    The result should be like this: Example1)사이즈 옵션: S, M, L\n도착 예정일: 10/7(토)\n가장 유용한 리뷰: 여성·164cm/49kg·M 사이즈, "적당한 오버핏으로 길이도 키에 딱 맞아서 편하고 예쁘고 후드티 핏이 좋다" Example2)사이즈 옵션: M, L\n도착 예정일: 10/6(금)\n가장 유용한 리뷰: 여성·162cm/53kg·L 사이즈, "품과 길이가 모두 만족스럽다\""""

    simple_detail = gpt(delivery, size_options, best_review, simple_detail_template)

    return { 
        "simple_detail": simple_detail,
        }

# from apikey import apikey
# url = 'https://www.musinsa.com/app/goods/3056893'
# #url = 'https://www.musinsa.com/app/goods/3603803'
# print(SimpleDetail(apikey, url))