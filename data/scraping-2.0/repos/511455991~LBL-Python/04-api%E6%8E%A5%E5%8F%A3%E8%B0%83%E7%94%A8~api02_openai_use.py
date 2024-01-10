# encoding=utf-8

"""
openai的chatgpt接口调用
author：LBL
date:2023-2-24
"""
import os
import openai

class chatgpt(object):
    def __init__(self):
        OPENAI_API_KEY = "sk-Q90kG9AoqzRyYf9j5YRRT3BlbkFJuNNICmZHHse1xhXJ9Mz81"
        openai.api_key = os.getenv("OPENAI_API_KEY",OPENAI_API_KEY)


    def text(self):
        """ 使用chatgpt处理文本信息 """
        # 设置要提的问题
        prompt = "模仿鲁迅写法编写五百字描写饥荒时，一个小男孩被救下的场景"
        # prompt = "`openai.Completion.create()`函数能接受哪些参数的输入，分别有什么作用"
        # 设置参数，发送请求，获取响应
        response = openai.Completion.create(
            model = "text-davinci-003",     # 设置模型model = "text-davinci-003"是 chatGPT 文本处理模型。
            prompt = prompt,                # 设置提问内容
            temperature = 1,                # 设置生成文本的多样性和创意度在0-1
            max_tokens = 2048,              # 设置生成文本的最大长度，最大为2048
            n = 1,                          # 设置要生成的文本数量
            stop = None                     # 设置文本停止条件，当生成文本中包含这些条件之一停止生成。为一个字符串或列表
        )
        # 提取响应内容中文本答案
        print(response.choices[0].text)


# 程序主入口
if __name__ == "__main__":
    """ 
    openai的key需要自己有chatgpt账户然后生成：https://platform.openai.com/account/api-keys
    处理文本模型：
        model = "text-davinci-003"是 chatGPT 最常用的模型之一。
        prompt = "问题"
    处理代码错误信息：
        model="code-davinci-002",处理代码错误信息
        prompt="##### Fix bugs in the below function\n \n### Buggy Python\nimport Random\na = random.randint(1,12)\nb = random.randint(1,12)\nfor i in range(10):\n    question = \"What is \"+a+\" x \"+b+\"? \"\n    answer = input(question)\n    if answer = a*b\n        print (Well done!)\n    else:\n        print(\"No.\")\n    \n### Fixed Python",
    chatgpt的其他参数可以通过询问chatgpt获取：`openai.Completion.create()`函数能接受哪些参数的输入，分别有什么作用
     """
    chat = chatgpt()
    chat.text()


""" 结果：
落叶发出沙沙的响声，太阳隐藏在天边的乌云之后，一片漆黑的夜晚开始漫山遍野把整个世界都笼罩进去，那只是暂时的停顿。饥荒，该来的总会来，当晚风在听到这词时也跟着呼啸，乌云在看到饥荒时也将哭泣中笼罩下一片片灰蒙蒙的天空。

古老的村庄，荒芜一片，原本温暖的林间小路已陷入死寂，整个村庄就像被抛弃的孩子一样，陷入了绝望绝望中......

此时，从地底传来了一阵轰隆大声的爆炸音，伴随着那声爆炸，整座村庄将瞬间拉回到正常，昔日村庄里的景象又重新雕刻入人们的记忆在里面。一个年轻的男孩被爆破的声音吵醒了，整个村庄慢慢地开始恢复生活，乌云缓缓地散去，星星璀璨发光，月亮把银色的朦胧在身后舞出万丈光芒，照亮着这个男孩。

当男孩慢慢站起来时，一个可怕的力量笼罩着他，他落入一片无边的黑暗中，消失不见了。好在，一群福利军在这瞬间来到村庄，好心的士兵已经把这个男孩救出来，他也伴随着欢呼中被带给了人们这片久未看到的希望。

落叶和福利军的鼓声，把整个希望的梦想都深深的印入这个男孩的心。感谢大家，也感谢这个男孩，他使我们相信，即使饥荒笼罩着他们，也有重新开始盼望的价值。希望我们能以这个男孩为榜样，跟随他走出饥荒，一起走出希望，一起找寻梦想给当下的坚强力量。
"""


