import openai
from backend import settings 


openai.api_key = settings.OPENAI_API_KEY


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role":"user", "content":prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of
    )
    return response.choices[0].message["content"]


# ai问答
def qa_ai(text):
    try:
        prompt = f"""
        对下方的被三反引号分隔的文本进行答复。
        \n 文本:```{text}```
        """
        print("答复中...")
        response = get_completion(prompt)
        return {'text': response, 'success': True, 'error': ''}
    except Exception as e:
        return {'text': '', 'success': False, 'error': e}        



# 翻译   
def translate(text, type):
    if type == 'trans2chn':
        content = f"""请将下列被三反引号分隔的文本翻译成中文，\n 文本:```{text}```"""
    elif type == 'trans2en':
        content = f"""请将下列被三反引号分隔的文本翻译成英文，\n 文本:```{text}```"""
    try:
        print("翻译中...")
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user', 'content': content}],
            temperature=0
        )
        corrected_text = response.choices[0].message["content"]
        return {'text': corrected_text, 'success': True, 'error': ''}
    except Exception as e:
        return {'text': '', 'success': False, 'error': e}






# 总结
def summary(text):
    try:
        prompt = f"""
        对下方的被三反引号分隔的文本进行总结，其中，总结的文本不超过30字。
        \n 文本:```{text}```
        """
        print("总结中...")
        response = get_completion(prompt)
        return {'text': response, 'success': True, 'error': ''}
    except Exception as e:
        return {'text': '', 'success': False, 'error': e}        


# 续写
def continueText(text):
    try:
        prompt = f"""
        对下方的被三反引号分隔的原文本进行续写，要求：续写文本和原文本的主题和情绪一致，且续写文本字数在120到300字之间。   
        \n 原文本:```{text}```
        """
        print("续写中..")
        response = get_completion(prompt)
        return {'text': response, 'success': True, 'error': ''}
    except Exception as e:
        return {'text': '', 'success': False, 'error': e}        

# 办公邮件
def emailText(text):
    try:
        prompt = f"""
        将下方的被三反引号分隔的原文本改写为邮件形式， \
        其中，该邮件用于办公和商务场景，语气正式，字数和原本文的字数相差不超过70字。\
        返回的格式为HTML格式，每一段被div标签包裹，段落之间用换行符分隔。\
        \n 原文本:```{text}```
        """
        print("写邮件中..")
        response = get_completion(prompt)
        return {'text': response, 'success': True, 'error': ''}
    except Exception as e:
        return {'text': '', 'success': False, 'error': e}        

# 校对更正
def proofread(text):
    try:
        prompt = f"""
        对下方的被三反引号分隔的原文本进行校对，\
        需要对有错别字和语法错误的地方进行标识，标识的方式为：用span标签包裹，且style属性为"color:red;text-decoration:line-through"。\
        返回信息中，保留原文本所有的错误内容，紧跟错误内容 提供正确的写法。
        返回信息的分段方式和原文本一致。返回信息用HTML格式，每一段用div标签包裹。 \
        \n 原文本:```{text}```
        """
        print("更正中..")
        response = get_completion(prompt)
        return {'text': response, 'success': True, 'error': ''}
    except Exception as e:
        return {'text': '', 'success': False, 'error': e}        


# 会议纪要
def meetingText(text):
    try:
        prompt = f"""
        将下方的被三反引号分隔的原文本改写为会议纪要， \
        其中，会议纪要的元素包括会议主题、时间、地点、参会人员、会议内容、会议结论。\
        每一个元素为一段，每段的格式为：元素名+冒号+元素内容。\
        元素名分别为：会议主题、时间、地点、参会人员、会议内容、会议结论。\
        如果没有某个元素，该元素的内容为空。\
        返回的格式为HTML格式，每一段都被div标签包裹，段落之间用换行符分隔。每个元素的元素名用<h2>标签包裹。元素内容的开始位置先加两个空格后再写内容  \
        \n 原文本:```{text}```
        """
        print("写会议纪要中..")
        response = get_completion(prompt)
        return {'text': response, 'success': True, 'error': ''}
    except Exception as e:
        return {'text': '', 'success': False, 'error': e}       


