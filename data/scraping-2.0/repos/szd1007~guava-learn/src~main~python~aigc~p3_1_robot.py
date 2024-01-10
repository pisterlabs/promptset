
import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")
COMPLETION_MODEL = "text-davinci-003"

prompt = '请你用朋友的语气回复给到客户，并称他为“亲”，他的订单已经发货在路上了，预计在3天之内会送达，订单号2021AEDG，我们很抱歉因为天气的原因物流时间比原来长，感谢他选购我们的商品。'

prompt_emotion_base = """
判断一下用户的评论情感上是正面的还是负面的
评论：买的银色版真的很好看，一天就到了，晚上就开始拿起来完系统很丝滑流畅，做工扎实，手感细腻，很精致哦苹果一如既往的好品质
情感：正面
评论：随意降价，不予价保，服务态度差
情感：负面
"""
prompt_emotion_good = """
评论：外形外观：苹果审美一直很好，金色非常漂亮拍照效果：14pro升级的4800万像素真的是没的说，太好了，运行速度：苹果的反应速度好，用上三五年也不会卡顿的，之前的7P用到现在也不卡其他特色：14pro的磨砂金真的太好看了，不太高调，也不至于没有特点，非常耐看，很好的
情感：
"""


prompt_emotion_bad = """
评论：信号不好电池也不耐电不推荐购买
情感：
"""

def get_response(prompt):
    completions = openai.Completion.create (
        engine=COMPLETION_MODEL,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=1.0,
    )
    message = completions.choices[0].text
    return message

print(prompt_emotion_good)
print(get_response(prompt_emotion_base + prompt_emotion_good))

print(prompt_emotion_bad)
print(get_response(prompt_emotion_base + prompt_emotion_bad))
