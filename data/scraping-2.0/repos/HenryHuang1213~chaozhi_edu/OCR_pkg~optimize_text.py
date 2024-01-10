import os

import openai


def get_completion(text, model="gpt-4"):
    json_format = """{
    "文章题目": <文章题目，如果没有，则返回无>,
    "文章正文": <正文内容>
}   
    """

    prompt = f"""
    下面是一段从ocr识别小学生手写中文作文得到的内容，用<<< >>>包含在内
    在ocr识别时，把页码也包含在内了，请把下面的ocr识别后的文字，去除你认为的非作文本身的内容
    同时，除了页面信息，可能也会包含教师的批注信息，需要删除
    另外，文字的分行也有错误，需要调整一下
    文章需要按照学生的想法正常分段
    不要对作文原文的内容有任何改动
    保留ocr识别后的文字，不要纠正错别字，不要补充额外的字
    如果识别正文前存在作文题目，请单独提取
    <<<{text}>>>
    
    请用如下JSON格式返回结果：
    {json_format}
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    openai.organization = "org-YtXb1vm6BeYmYEPTIAG61m59"
    openai.api_key_path = os.path.join(dir_path, '../api.key')
    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

