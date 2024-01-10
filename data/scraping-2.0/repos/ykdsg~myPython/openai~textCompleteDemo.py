import os
import openai
from dotenv import load_dotenv

# openai textComplete例子：https://platform.openai.com/docs/quickstart/build-your-application

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def index(animal):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=generate_prompt(animal),
        temperature=0.6,
    )
    result = response.choices[0].text
    print("result:", result)


def generate_prompt(animal):
    return """Suggest three names for an animal that is a superhero.
Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:""".format(
        animal.capitalize()
    )


COMPLETION_MODEL = "text-davinci-003"


def get_response(prompt, temperature=1.0):
    completions = openai.Completion.create(
        engine=COMPLETION_MODEL,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=temperature,
    )
    message = completions.choices[0].text
    return message


# 情感分类
def emotion():
    prompts = """判断一下用户的评论情感上是正面的还是负面的
    评论：买的银色版真的很好看，一天就到了，晚上就开始拿起来完系统很丝滑流畅，做工扎实，手感细腻，很精致哦苹果一如既往的好品质
    情感：正面

    评论：随意降价，不予价保，服务态度差
    情感：负面
    """
    good_case = prompts + """
    评论：外形外观：苹果审美一直很好，金色非常漂亮
    拍照效果：14pro升级的4800万像素真的是没的说，太好了，
    运行速度：苹果的反应速度好，用上三五年也不会卡顿的，之前的7P用到现在也不卡
    其他特色：14pro的磨砂金真的太好看了，不太高调，也不至于没有特点，非常耐看，很好的
    情感：
    """
    print(get_response(good_case))

    bad_case = prompts + """
    评论：信号不好电池也不耐电不推荐购买
    情感
    """
    print(get_response(bad_case))


if __name__ == '__main__':
    # index("horse")
    emotion()
