import openai
import time
import re
import cohere

# 在环境变量中设置OpenAI API密钥
# import requests

openai.api_key = "xxx"
cohere_api_key = "xxx"
openai.api_base = "xxx"


def get_context(prompt: str) -> str:
    """
    :param prompt: 要输入的提示
    :return: 获取到的结果
    """
    try:
        co = cohere.Client(cohere_api_key)
        question = "请判断文章标题为'" + prompt + "'是医学中[内科，外科，儿科，妇产科，骨科，影像科，其他]中的哪个领域，用这是医学中的...领域来回答"
        response = co.generate(
            model='command-xlarge-nightly',
            # model='medium',
            prompt=question,
            max_tokens=100,
            temperature=0.7,
            stop_sequences=["--"]
        )
        context = response.generations[0].text
    except Exception as e:
        print(e)
        context = "其他"
    finally:
        return context


# 提问代码
def get_category(title: str):
    # 你的问题
    try:
        question = "请判断标题为'" + title + "'的文献是医学中[内科，外科，儿科，妇产科，骨科，影像科，其他]中的哪个领域，用'这是医学中的...领域来回答'"
        prompt = question
        # print(prompt)
        # 调用 ChatGPT 接口
        model_engine = "text-davinci-003"
        create = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=1024, n=1, stop=None,
                                          temperature=0.5, )
        completion = create

        response = completion.choices[0].text
        # print(response)
        rule = r'这是医学中的(.*?)领域'
        category = re.findall(rule, response)[0].strip()
        return category
    except Exception as e:
        # print(e)
        return e


if __name__ == '__main__':
    title = "Annoyance to different noise sources is associated with atrial fibrillation"
    start = time.time()
    # response = get_context(prompt=title)
    response = get_category(title)
    end = time.time()
    # print("标题是：" + title)
    print(response)
    # print("耗时", end - start)
