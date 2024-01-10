from cache_utils import cache
import openai
import deepl
import os

from getData.get_reviews import GetReviews

def gpt(up_reviews, worst_reviews, template):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.9,
        messages=[{"role": "system", "content": template},
                  {"role": "user", "content": f"{up_reviews} {worst_reviews}"}],
        max_tokens=500  # 원하는 최대 토큰 수를 설정
    )

    return response.choices[0].message.content.strip()

def ReviewSumm(apikey, id):
    openai.api_key = apikey

    cache_key = f"{id}_reviews"
    reviews = cache.get(cache_key)
    if reviews is None:
        url = 'https://www.musinsa.com/app/goods/' + id
        up_reviews, worst_reviews = GetReviews(url)
        if up_reviews:
            auth_key = os.getenv("DEEPL_API_KEY")
            translator = deepl.Translator(auth_key=auth_key)
            up_reviews_eng = translator.translate_text(up_reviews, target_lang="EN-US")
            worst_reviews_eng = translator.translate_text(worst_reviews, target_lang="EN-US")
        reviews = {"up_reviews":up_reviews_eng, "worst_reviews":worst_reviews_eng}
        cache[cache_key] = reviews

    up_reviews = reviews["up_reviews"]
    worst_reviews = reviews["worst_reviews"]

    if up_reviews:
        review_sum_template = """You are the helpful agent that summarize best reviews and worst reviews as 2~3 sentences separately. You must return summaries in Korean.
        The result should be like this: Example)[Useful Reviews]\n- I like Sibori because it's sturdy.\n- The combination of black and pink is cute.\n- It's popular for its beautiful design and colors.\n\n[Low Rated Reviews]\n- The quality and fit are good, but the neck part is slightly small.\n- The sleeves are short.\n- It would be nice if there were more size options."""
        # The result should be like this: Example)[Useful Reviews]\n- 시보리가 짱짱해서 마음에 든다.\n- 블랙과 핑크 조합이 귀엽다.\n- 예쁜 디자인과 색상으로 인기가 많다.\n[Low Rated Reviews]\n- 품질과 핏이 좋지만 목부분이 살짝 감긴다.\n- 팔 길이가 짧다.\n- 사이즈 종류가 다양했으면 좋겠다.
        result = gpt(up_reviews_eng, worst_reviews_eng, review_sum_template)
        result_ko = translator.translate_text(result, target_lang="KO")
        reviewSumm = "구매자들의 리뷰를 유용한 리뷰와 평점이 낮은 리뷰로 나눠서 요약해드리겠습니다.\n" + str(result_ko)
    else:
        reviewSumm = "리뷰가 존재하지 않습니다."

    return { 
            "review_summ": reviewSumm
        }

# apikey = os.getenv("OPENAI_API_KEY")
# url = 'https://www.musinsa.com/app/goods/3056893'
# url = 'https://www.musinsa.com/app/goods/3603803'
# print(ReviewSumm(apikey, url))