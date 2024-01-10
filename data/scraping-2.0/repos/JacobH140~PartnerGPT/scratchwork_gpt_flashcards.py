import utils
import ast
import os
from dotenv import load_dotenv
from secret_openai_apikey import api_key
import openai
import pandas as pd



# get chatgpt's response
#os.environ['OPENAI_API_KEY'] = api_key 
openai.api_key = api_key
load_dotenv()

arbitrary_note_prompt = """
You are a helpful AI Chinese language teacher, helping a student create Anki flashcards. I will give you a word in one of english, simplfied, or traditional Chinese. Please respond with the following information, formatted as a python dictionary (quoted strings are the dictionary keys).

- "简体字Simplified" : the simplified Chinese word.
- "繁体字 Traditional" : the traditional Chinese word.
- "英文English" : the English translation. Can be as long as you want for nuanced words, but be concise and clear.
- "Simplification Process (GPT Estimate)" : the process of simplifying the traditional Chinese word to the simplified Chinese word, say "None" if the word is already simplified.
- "Vocab pinyin" : the pinyin of the Chinese word.
- "Etymology — GPT Conjecture" : etymology of the whole word... how do the individual characters' meaning contribute to the whole word's meaning? You may choose to analyze one or both of the simplified and traditional versions.
- "Categories of Characters — GPT Conjecture" : The type of each character involved, perhaps one of 象形字,  形声字, 指事字,  会意字,  转注字, 假借字. Explain your answer. If you claim a character is a 形声字, you should check that the phonetic component matches the pinyin you provided before. If not, it is not 形声字 and you should think again about this!
- "例句 Example sentence simplified" : An example sentence at the HSK3 level, in simplified Chinese. Disclude translation/pinyin.
- "例句 Example sentence traditional" : The same sentence as above in traditional Chinese.
- "例句 Example sentence pinyin" : The same sentence as above in pinyin.
- "例句 Example sentence translation" : The same sentence as above in English.
- "Related words" : Words commonly used alongside the word, and description or example of the relation.
- "同义词/同義詞Synonyms" : Synonyms. Include pinyin and translation.
- "反义词/反義詞Antonyms" : Antonyms. Include pinyin and translation.
- 量词/量詞Classifier(s) : Measure words, if relevant. Include pinyin.
- "Usages" : Any words, common phrases, idioms, etc. that use this word. Include pinyin and translation. For example, for 的 the response could be 有的时候, 别的, 是她做的，什么的. This is NOT just a place to add example sentences!

Thanks!"""



higher_temp_prompt = """
- "例句 Example sentence simplified" : An example sentence at the HSK3 level, in simplified Chinese. Disclude translation/pinyin.
- "例句 Example sentence traditional" : The same sentence as above in traditional Chinese.
- "例句 Example sentence pinyin" : The same sentence as above in pinyin.
- "例句 Example sentence translation" : The same sentence as above in English.
- "Related words" : Words commonly used alongside the word, and description or example of the relation.
- "同义词/同義詞Synonyms" : Synonyms. Include pinyin and translation.
- "反义词/反義詞Antonyms" : Antonyms. Include pinyin and translation.
- 量词/量詞Classifier(s) : Measure words, if relevant. Include pinyin.
- "Usages" : Any words, common phrases, idioms, etc. that use this word. Include pinyin and translation. For example, for 的 the response could be 有的时候, 别的, 是她做的，什么的. This is NOT just a place to add example sentences!

Thanks!"""

list = """钓 diao4
顶 ding3
动画片 dong4hua4pian4
冻 dong4
洞 dong4
豆腐 dou4fu5
逗 dou4
独立 du2li4
独特 du2te4
度过 du4guo4
断 duan4
堆 dui1
对比 dui4bi3
对待 dui4dai4
对方 dui4fang1
对手 dui4shou3
对象 dui4xiang4
兑换 dui4huan4
吨 dun1
蹲 dun1
顿 dun4
多亏 duo1kui1
多余 duo1yu2
朵 duo3
躲藏 duo3cang2
恶劣 e4lie4
耳环 er3huan2
发表 fa1biao3
发愁 fa1//chou2
发达 fa1da2
发抖 fa1dou3
发挥 fa1hui1
发明 fa1ming2
发票 fa1piao4
发言 fa1yan2
罚款 fa2kuan3
法院 fa3yuan4
翻 fan1
繁荣 fan2rong2
反而 fan3er2
反复 fan3fu4
反应 fan3ying4
反映 fan3ying4
反正 fan3zheng4
范围 fan4wei2
方 fang1
方案 fang1an4
方式 fang1shi4
妨碍 fang2ai4
仿佛 fang3fu2
非 fei1
肥皂 fei2zao4
废话 fei4hua4
分别 fen1bie2
分布 fen1bu4
分配 fen1pei4
分手 fen1//shou3
分析 fen1xi1
纷纷 fen1fen1
奋斗 fen4dou4
风格 feng1ge2
风景 feng1jing3
风俗 feng1su2
风险 feng1xian3
疯狂 feng1kuang2
讽刺 feng3ci4
否定 fou3ding4
否认 fou3ren4
扶 fu2
服装 fu2zhuang1
幅 fu2
辅导 fu3dao3
妇女 fu4nü3
复制 fu4zhi4
改革 gai3ge2
改进 gai3jin4
改善 gai3shan4
改正 gai3zheng4
盖 gai4
概括 gai4kuo4
概念 gai4nian4
干脆 gan1cui4
干燥 gan1zao4
赶紧 gan3jin3
赶快 gan3kuai4
感激 gan3ji1
感受 gan3shou4
感想 gan3xiang3
干活儿 gan4huo2r5
钢铁 gang1tie3
高档 gao1dang4
高级 gao1ji2
搞 gao3
告别 gao4bie2
格外 ge2wai4
隔壁 ge2bi4
个别 ge4bie2
个人 ge4ren2
个性 ge4xing4
各自 ge4zi4
根 gen1
根本 gen1ben3
工厂 gong1chang3
工程师 gong1cheng2shi1
工具 gong1ju4
工人 gong1ren5
工业 gong1ye4
公布 gong1bu4
公开 gong1kai1
公平 gong1ping2
公寓 gong1yu4
公元 gong1yuan2
公主 gong1zhu3
功能 gong1neng2
恭喜 gong1xi3
贡献 gong4xian4
沟通 gou1tong1
构成 gou4//cheng2
姑姑 gu1gu5
姑娘 gu1niang5
古代 gu3dai4
古典 gu3dian3
股票 gu3piao4
骨头 gu3tou5
鼓舞 gu3wu3
鼓掌 gu3//zhang3
固定 gu4ding4
挂号 gua4//hao4
乖 guai1
拐弯 guai3wan1
怪不得 guai4bu5de5
关闭 guan1bi4
观察 guan1cha2
观点 guan1dian3
观念 guan1nian4
官 guan1
管子 guan3zi5
冠军 guan4jun1
光滑 guang1hua2
光临 guang1lin2
光明 guang1ming2
光盘 guang1pan2
广场 guang3chang3
广大 guang3da4
广泛 guang3fan4
归纳 gui1na4
规矩 gui1ju5
规律 gui1lü4
规模 gui1mo2
规则 gui1ze2
柜台 gui4tai2
滚 gun3
锅 guo1
国庆节 Guo2qing4jie2
国王 guo2wang2
果然 guo3ran2
果实 guo3shi2
过分 guo4fen4
过敏 guo4min3
过期 guo4//qi1
哈 ha1
海关 hai3guan1
海鲜 hai3xian1
喊 han3
行业 hang2ye4
豪华 hao2hua2
好客 hao4ke4
好奇 hao4qi2
合法 he2fa3
合理 he2li3
合同 he2tong5
合影 he2ying3
合作 he2zuo4
何必 he2bi4
何况 he2kuang4
和平 he2ping2
核心 he2xin1
恨 hen4
猴子 hou2zi5
后背 hou4bei4
后果 hou4guo3
呼吸 hu1xi1
忽然 hu1ran2
忽视 hu1shi4
胡说 hu2shuo1
胡同 hu2tong4
壶 hu2
蝴蝶 hu2die2
糊涂 hu2tu5
花生 hua1sheng1
划 hua4
华裔 hua2yi4
滑 hua2
化学 hua4xue2
话题 hua4ti2
怀念 huai2nian4
怀孕 huai2//yun4
缓解 huan3jie3
幻想 huan4xiang3
慌张 huang1zhang1
黄金 huang2jin1
灰 hui1
灰尘 hui1chen2
灰心 hui1//xin1
挥 hui1
恢复 hui1fu4
汇率 hui4lü4
婚礼 hun1li3
婚姻 hun1yin1
活跃 huo2yue4
火柴 huo3chai2
伙伴 huo3ban4
或许 huo4xu3
机器 ji1qi4
肌肉 ji1rou4
基本 ji1ben3
激烈 ji1lie4
及格 ji2//ge2
极其 ji2qi2
急忙 ji2mang2
急诊 ji2zhen3
集合 ji2he2
集体 ji2ti3
集中 ji2zhong1
计算 ji4suan4
记录 ji4lu4
记忆 ji4yi4
纪录 ji4lu4
纪律 ji4lü4
纪念 ji4nian4
系领带 ji4ling3dai4
寂寞 ji4mo4
夹子 jia1zi5
家庭 jia1ting2
家务 jia1wu4
家乡 jia1xiang1
嘉宾 jia1bin1
甲 jia3
假如 jia3ru2
假设 jia3she4
假装 jia3zhuang1
价值 jia4zhi2
驾驶 jia4shi3
嫁 jia4
坚决 jian1jue2
坚强 jian1qiang2
肩膀 jian1bang3
艰巨 jian1ju4
艰苦 jian1ku3
兼职 jian1zhi2
捡 jian3
剪刀 jian3dao1
简历 jian3li4
简直 jian3zhi2
建立 jian4li4
建设 jian4she4
建筑 jian4zhu4
健身 jian4//shen1
键盘 jian4pan2
讲究 jiang3jiu5
讲座 jiang3zuo4
酱油 jiang4you2
交换 jiao1huan4
交际 jiao1ji4
交往 jiao1wang3
浇 jiao1
胶水 jiao1shui3
角度 jiao3du4
狡猾 jiao3hua2
教材 jiao4cai2
教练 jiao4lian4
教训 jiao4xun5
阶段 jie1duan4
结实 jie1shi5
接触 jie1chu4
接待 jie1dai4
接近 jie1jin4
节省 jie2sheng3
结构 jie2gou4
结合 jie2he2
结论 jie2lun4
结账 jie2//zhang4
戒 jie4
戒指 jie4zhi5
届 jie4
借口 jie4kou3
金属 jin1shu3
尽快 jin3kuai4
尽量 jin3liang4
紧急 jin3ji2
谨慎 jin3shen4
尽力 jin4//li4
进步 jin4bu4
进口 jin4//kou3
近代 jin4dai4
经典 jing1dian3
经商 jing1//shang1
经营 jing1ying2
精力 jing1li4
精神 jing1shen2
酒吧 jiu3ba1
救 jiu4
救护车 jiu4hu4che1
舅舅 jiu4jiu5
居然 ju1ran2
桔子 ju2zi5
巨大 ju4da4
具备 ju4bei4
具体 ju4ti3
俱乐部 ju4le4bu4
据说 ju4shuo1
捐 juan1
决赛 jue2sai4
决心 jue2xin1
角色 jue2se4
绝对 jue2dui4
军事 jun1shi4
均匀 jun1yun2
卡车 ka3che1
开发 kai1fa1
开放 kai1fang4
开幕式 kai1mu4shi4
开水 kai1shui3
砍 kan3
看不起 kan4bu5qi3
看望 kan4wang4
靠 kao4
颗 ke1
可见 ke3jian4
可靠 ke3kao4
可怕 ke3pa4
克 ke4
克服 ke4fu2
刻苦 ke4ku3
客观 ke4guan1
课程 ke4cheng2
空间 kong1jian1
空闲 kong4xian2
控制 kong4zhi4
口味 kou3wei4
夸 kua1
夸张 kua1zhang1
会计 kuai4ji5
宽 kuan1
昆虫 kun1chong2
扩大 kuo4da4
辣椒 la4jiao1
拦 lan2
烂 lan4
朗读 lang3du2
劳动 lao2dong4
劳驾 lao2//jia4
老百姓 lao3bai3xing4
老板 lao3ban3
老婆 lao3po5
老实 lao3shi5
老鼠 lao3shu3
姥姥 lao3lao5
乐观 le4guan1
雷 lei2
类型 lei4xing2
冷淡 leng3dan4
厘米 li2mi3
离婚 li2//hun1
梨 li2
理论 li3lun4
理由 li3you2
力量 li4liang5
立即 li4ji2
立刻 li4ke4
利润 li4run4
利息 li4xi1
利益 li4yi4
利用 li4yong4
连忙 lian2mang2
连续 lian2xu4
联合 lian2he2
恋爱 lian4ai4
良好 liang2hao3
"""


if __name__ == "__main__":

    gpt_model = "gpt-3.5-turbo"
    success = False
    print(utils.get_chatgpt_response([{f"role":"user", "content":f"remove all of the pinyin from the following list: {list}, please. then show me the resulting list."}], temperature=0))

    #while not success:
    #    try:
    #        first_user_prompt = """The word is "in english, when i am seeing if some functionality works (while coding, etc), i use the word 'test'. what is the counterpart in chinese?". Format your response as a python dictionary (where quoted strings are the dictionary's keys)."""
    #        messages = [{"role": "system", "content": f"{arbitrary_note_prompt}"},{"role": "user", "content": f"{first_user_prompt}"}]
    #        first_response = utils.get_chatgpt_response(messages, temperature=0.7, model=gpt_model)
    #        #messages = utils.update_chat(messages, "assistant", first_response)
    #        #messages.append({"role": "user", "content": f"Continue the dictionary with {higher_temp_prompt}"})
    #        #second_response = utils.get_chatgpt_response(messages, temperature=1, model=gpt_model)
    #        print("System: ", arbitrary_note_prompt)
    #        print("User: ", first_user_prompt)
    #        print("Response: ", first_response)
    #        output_dict = ast.literal_eval(first_response)
    #        success = True
    #    except Exception as e:
    #        print(e)
    #        print("Trying again, cGPT gave incorrectly formatted output")
#
#
    #print(output_dict)