
import getpass
import os
import openai
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from llm_utils.read_config import load_config



config = load_config() 
#os.environ["https_proxy"] = "http://localhost:7890"
os.environ["OPENAI_API_BASE"] = config['openai_base']
os.environ["OPENAI_API_KEY"] = config['openai_api_key']



def base_chat():
    openai.api_key = config['openai_api_key']
    openai.api_base = config['openai_base']
    
    file = open("./llm_utils/prompt.txt", "r", encoding="utf-8")
    
    # 读取文件内容 = file.read()
    prompt = file.read()
    # 关闭文件
    file.close()
    
    #prompt = f"请你说一段土味情话，在说之前请先学习下面的土味情话的技巧，但不要出现类似的内容：1、“孔子.孟子.老子，你知道你最适合当什么子吗?”“不知道。”“我的妻子。” 　　2、“你猜我爱喝酒还是爱打王者” “爱打王者吧?” “不，爱你” 　3、你知道吗?你最近一直怪怪的怪好看的　　4、“人生有两次幸运就行”说说?“一次遇见你，一次睡到底” 　　5、我的手被划了一口子，你也划一下，这样我们就是两口子。　　6、最近老是头晕你知道是什么原因吗不知道因为爱情使人头晕。　　7、“我的胸口感觉好闷啊”怎么回事?生病了吗?“不是，因为你卡在我的心口上” 　　8、遇见喜欢的人，就像浩劫余生，漂流过沧海，终见陆地。　　9、“你能不能闭嘴”“我没有说话啊”“那为什么我满脑子都是你的声音” 　　10、这是牛肉，这是猪肉，你是我的心头肉。　　11、“我呢，喜欢一个人”然后呢“然后他给我发了一个然后呢” 　　12、“你知道你和星星有什么区别吗?”“有什么区别”“星星在天上，而你在我心里。” 　　13、“你对我笑一下。你不对我笑一下我晚上怎么睡得着。” 　　14、“忙了一天，我想睡觉了。”睡吧“你过来跟我一起睡吧” 　　15、我生来执拗，爱憎分明，从见到你的第一眼起，我的眸子里，就全是清澈的你，所以我喜欢你，不是一时兴起，也不是心口不　　16、现在几点了?(12点)不，是我们幸福的起点　　17、这是手背，这是脚背，这是我的宝贝。　　18、你闻到什么味道了吗?没有啊。怎么你一出来空气就甜炸了啊。 　　19、“我想搬家”搬哪?“搬到你心里去” 　　20、我最近一直在找一家店。什么店?你的来电。    你的回答只用输出土味情话即可"
    print(prompt)
    # gpt-3.5-turbo-0301     
    completion = openai.ChatCompletion.create(
      model='gpt-3.5-turbo-0301', 
      messages=[
        {"role": "user", "content":prompt}
      ]
    )

    #print(completion)    
    
    #print("completion.usage", completion.usage) 
    print(completion.choices[0].message['content'])  
    return completion.choices[0].message['content']