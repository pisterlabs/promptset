# encoding:utf-8
# Note: The openai-python library support for Azure OpenAI is in preview.
import json
import os
import time

import openai

os.environ["OPENAI_API_KEY"] = "33e8f0c860bc4109825496444bbfed3e"
openai.api_type = "azure"
openai.api_base = "https://community-openai-34.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")


#######################
## 生成问题
#######################
def get_question(content_file, fw):
    content = json.load(open(content_file, 'r', encoding='utf-8'))
    for c in content:
        for chapter in c:
            for k1 in c[chapter]:
                print(k1)
                for k2 in c[chapter][k1]:
                    # print(c[chapter][k1][k2].keys())
                    # time.sleep(10)
                    if "question" not in c[chapter][k1][k2].keys():
                        print(k2)
                        print(c[chapter][k1][k2])
                        inputs = "针对’" + k2 + "‘进行提问，注意只能提一个问题，给出问题和答案！问题需要尽量详细地考察以下内容\n" + \
                                 c[chapter][k1][k2]["text"]
                        response = openai.ChatCompletion.create(
                            engine="gpt35-34",
                            messages=[
                                {"role": "system",
                                 "content": "现在你是一个提问者，针对某个主题进行提问，给出问题和答案。注意只能提一个问题。"},
                                {"role": "user", "content": inputs[:5500]},
                            ],
                            temperature=1.0,
                            max_tokens=800,
                            top_p=1,
                            n=5,
                            frequency_penalty=0,
                            presence_penalty=0,
                            stop=None)
                        print(response["choices"][0]["message"]["content"])
                        print(response["choices"][1]["message"]["content"])
                        print(response["choices"][2]["message"]["content"])
                        print(response["choices"][3]["message"]["content"])
                        print(response["choices"][4]["message"]["content"])
                        c[chapter][k1][k2]["questions"] = \
                            [response["choices"][0]["message"]["content"],
                             response["choices"][1]["message"]["content"],
                             response["choices"][2]["message"]["content"],
                             response["choices"][3]["message"]["content"],
                             response["choices"][4]["message"]["content"]]
                        json_data = {
                            "text": c[chapter][k1][k2]["text"],
                            "questions": c[chapter][k1][k2]["questions"]
                        }
                        fw.write(json.dumps(json_data, ensure_ascii=False) + '\n')
                        time.sleep(30)

# content_file = "/Users/zhanglin/Desktop/LM_as_Evaluator/data/taxonomy/history_qa_v1.json"
# content = json.load(open(content_file, 'r', encoding='utf-8'))
#
# fw = open("history_questions_candidates.txt", 'a', encoding='utf-8')
# for c in content:
#     for chapter in c:
#         for k1 in c[chapter]:
#             print(k1)
#             for k2 in c[chapter][k1]:
#                 # print(c[chapter][k1][k2].keys())
#                 # time.sleep(10)
#                 if "question" not in c[chapter][k1][k2].keys():
#                     print(k2)
#                     print(c[chapter][k1][k2])
#                     inputs = "针对’" + k2 + "‘进行提问，注意只能提一个问题，给出问题和答案！问题需要尽量详细地考察以下内容\n" + \
#                              c[chapter][k1][k2]["text"]
#                     response = openai.ChatCompletion.create(
#                         engine="gpt35-34",
#                         messages=[
#                             {"role": "system",
#                              "content": "现在你是一个提问者，针对某个主题进行提问，给出问题和答案。注意只能提一个问题。"},
#                             {"role": "user", "content": inputs[:5500]},
#                         ],
#                         temperature=1.0,
#                         max_tokens=800,
#                         top_p=1,
#                         n=5,
#                         frequency_penalty=0,
#                         presence_penalty=0,
#                         stop=None)
#                     print(response["choices"][0]["message"]["content"])
#                     print(response["choices"][1]["message"]["content"])
#                     print(response["choices"][2]["message"]["content"])
#                     print(response["choices"][3]["message"]["content"])
#                     print(response["choices"][4]["message"]["content"])
#                     c[chapter][k1][k2]["questions"] = \
#                         [response["choices"][0]["message"]["content"],
#                          response["choices"][1]["message"]["content"],
#                          response["choices"][2]["message"]["content"],
#                          response["choices"][3]["message"]["content"],
#                          response["choices"][4]["message"]["content"]]
#                     json_data = {
#                         "text": c[chapter][k1][k2]["text"],
#                         "questions": c[chapter][k1][k2]["questions"]
#                     }
#                     fw.write(json.dumps(json_data, ensure_ascii=False) + '\n')
#                     time.sleep(30)
#
# with open("history_candidates.json", 'w', encoding='utf-8') as f:
#     f.write(json.dumps(content, ensure_ascii=False))

# input = "针对’当代中国的法治与精神文明建设‘进行提问，给出问题和答案。问题需要尽量详细地考察以下内容：" \
#         "近代西方的法律与教化 在罗马法的基础上，英 国和法国分别发展了英美法 系和大陆法系。 学习聚焦 希腊克里特岛有着辉煌的古代文明，这里的一些 城邦，很早就有习惯法，也出现了成文法。在当地的 一处遗址中，发现了公元前7世纪的石刻，上面的铭 文记载了有关法律的内容。这就是早期的成文法。 ▲克里特岛上的一处遗址 ▲铭文摹本 ▲《十二铜表法》在罗马街头颁布 时人们围观的情形 近代西方法律制度的渊源及发展 为了缓和平民和贵族的矛盾，公元前450年左右，罗 马共和国颁布了《十二铜表法》。罗马帝国时期，随着统 治区域的扩大和人口的激增，法律制度更加完善。6世纪， 东罗马帝国 ①皇帝查士丁尼下令编纂的《罗马民法大全》， 是古罗马法律的最高成就，也是近代西方法律制度的渊源。 中古时期，各日耳曼王国在记载和整理日耳曼人部落 习惯法的基础上编纂了一批成文法，称为“日耳曼法”， 作为庄园法庭审判的依据。教会也根据基督教神学，制定 和颁布了教会法。11世纪以后，欧洲国家出现了研究和宣 传罗马法的运动，促进了罗马法的传播。 11世纪，诺曼底公爵征服英国，建立了诺曼王朝。为 了加强对地方的控制，王室设立法院，并派法官定期到各 地进行巡回审判。12世纪前后，建立在习惯法基础上、全 ①东罗马帝国又称拜占庭帝国。 国普遍适用的法律在英国逐渐形成，这就是普通法。13世 纪，英国通过《大宪章》，确立了法律至上和王权有限的 原则。“光荣革命”后，英国确立了君主立宪制，法律体系 更加完善。美国等很多国家在学习英国法律的基础上制定 了本国法律，它们构成了普通法系，也称“英美法系”。 13世纪以后，随着王权的加强，法国统一法律的步伐 加快，建立在罗马法基础上的法律体系日益成熟。1789年， 法国爆发大革命。此后，法国在启蒙思想和大革命的影响 下，制定了一系列法律。1804年，拿破仑签署法令，颁布 了《法国民法典》，它与此后制定的四部法典一起被统称为 《拿破仑法典》。《拿破仑法典》与此前颁布的法律，构成了 法国的成文法体系，最终确立了法国的资产阶级法律制度。 后来，逐渐形成了以罗马法为基础、以《法国民法典》为 代表的世界性法律体系，称为“大陆法系”或“民法系”。 近代西方法律制度的基本特征 近代以来，西方各国在继承传统法律思想的基础上， 融合了启蒙思想家们提出的思想主张，制定了各自的法律 制度。这些法律制度从理论上看，包含着一些共同的基本 特征。 英美法系和大陆法系的不同 英美法系以判例法为主要法律渊源，以遵循先例为基本原 则；法官的地位突出，当无先例可循时，法官可以创立先例， 也可以对先例作出新的解释。因此，英美法系国家的法律也被 称为“法官制定的法律”。英美法系主要涵盖英国、美国、加 拿大、澳大利亚、印度等国家和地区。 大陆法系以成文法为主要法律渊源，强调宪法的根本法地 位，法律体系比较完整，一般不承认判例的效力；明确立法和 司法的分工，法官的作用不太突出。大陆法系国家的代表是法 国、德国、意大利、日本等。 历史纵横 ▲《大宪章》原件 法国大革命对法律建设 有什么重要贡献？ 思考点 近代西方法律制度的基 本特征是立法和司法独立， 强调保障个人的权利。 学习聚焦 在国家权力结构层面上，坚持权力制衡、三权分立。 国家权力分为立法权、行政权和司法权。法律由议会制定， 行政机构在法律规定的框架内行使行政权，法院根据法律 独立掌握司法权。在法律内容上，注重保护个人权利，包 括生命权、自由权和财产权等。在司法实践过程中，坚持 程序公正和无罪推定。为了保证从立案到审理再到判决的 每个程序的公开公正，建立了律师制度和陪审团制度。独 立、专业的律师为被审判者提供辩护，可以减少法官对法 律的误读；从民众中产生陪审团，参与案件审理和判决， 使民众能够直接参与法律事务。无罪推定原则指的是所有 被审判者在判决之前都被视为无罪。 西方法律制度为资产阶级利益服务，存在着许多局限 性。它确认了私有财产制度，每个人财产的多少往往决定 着法律地位的高低。同时，对个人权利的认定也有逐渐改 进的过程。直到20世纪，黑人、原住民和妇女还在为享有 完全的公民权积极斗争。 宗教伦理与教化 392年，基督教成为罗马国教。476年，西罗马帝国灭 亡。在帝国废墟上建立的日耳曼人国家为了取得罗马人和 教会的支持，逐渐接受了基督教。基督教影响了中古时期 ▲林肯（右一）在法庭上辩护的情景 1836年，林肯成为律师。他在 诉讼活动中以敢于主持正义、熟练 运用法律而享有盛名，为以后当选 美国总统奠定了基础。 陪审团制度和律师制度的起源 陪审团制度最早可以追溯到古希腊罗马时期，但当时只适 用于奴隶主和自由民。古代的日耳曼人也留有“同侪裁决”的 遗风。12世纪，英国确立了陪审团制度。陪审团在法庭上聆听 证据，就事实问题作出决定；法官决定法律问题，判断某项证 据是否可以被引入审判。律师制度的起源也可以追溯到罗马时 期。资产阶级革命后，各国纷纷颁布法律，支持被告自己或聘 请律师辩护。1878年德国颁布的《国家律师法》，奠定了近代 律师制度的基础。 历史纵横 基督教的宗教伦理不仅 强化了教会对人们的控制， 也具有一定的社会教化功能。 学习聚焦 1764年7月，意大利人 贝卡里亚提出： 在法官判决之前，一个 人是不能被称为罪犯的。只要 还不能断定他已经侵犯了给 予他公共保护的契约，社会 就不能取消对他的公共保护。 —［意］贝卡里亚著， 黄风译《论犯罪与刑罚》 史料阅读 宗教改革后，西欧的基督教分裂为天主教和新教。新 教适应了原始积累时期新兴资产阶级的政治、经济诉求， 提出了一些新的主张。新教反对教皇权威，主张信徒通过 自己阅读《圣经》理解教义，还提倡节俭和积极入世的态 度，鼓励人们发财致富。但是，新教仍然坚持基督教的基 欧洲人的政治、经济和社会生活各个方面。教士们搜集和 抄录经典，保存了一些宝贵的古典文化，但他们更重要的 任务是宣讲教义。他们还开办学校，这些学校主要是宗教 学校，也有一些世俗学校。学校主要讲授宗教内容，也教 授算术、几何、天文、音乐、文法、修辞和逻辑。学习内 容虽然都以宗教为目的，但在教育和文化方面也发挥了重 要作用。人们的生老病死、婚丧嫁娶，基督教会都要介入， 几乎所有的节日都与基督教有关。教会尽管本身藏污纳垢， 但时刻不忘告诫人们必须孝敬父母，不许偷盗、奸淫、杀 人、贪恋别人的财物等，要求人们逆来顺受、忍受世间的 一切痛苦。基督教的宗教伦理和教化作用强化了教会对人 们的控制，深刻影响了人们的思想意识和日常行为。 宗教法庭 在基督教会的控制下，违背基督教伦理的行为往往为社会 所不容。教会不允许任何质疑声音的存在，甚至为此建立了宗 教法庭或宗教裁判所。1480年，西班牙建立了国王控制的宗教 法庭，以此来打击异己。到1820年，西班牙宗教法庭审判的 “异端”有30多万人，其中10多万人被判处火刑。 历史纵横 中世纪完全是从野蛮状态发展而来的。……在僧侣手中， 政治和法学同其他一切科学一样，不过是神学的分支，一切都 按照神学中适用的原则来处理。教会的教条同时就是政治信条， 圣经词句在各个法庭都具有法律效力。 ——［德］恩格斯《德国农民战争》，《马克思恩格斯文 集》第二卷 想一想：基督教在中古时期的欧洲发挥了什么样的作用？ 学思之窗 ▲宗教法庭 ▲中古时期的婚礼 本教义，束缚人们的行为，麻醉人们的思想。新教还排斥 其他教派，引起了多次宗教冲突，造成了重大的人员伤亡 和财产损失。一些对教义持有不同意见的人被斥为“异 端”，遭到迫害。例如，1553年，西班牙科学家塞尔维特 在日内瓦被加尔文派判处火刑。 路德战胜了虔信造成的奴役制，是因为他用信念造成的奴役制代替了它。他破除了对权威的 信仰，是因为他恢复了信仰的权威。他把僧侣变成了世俗人，是因为他把世俗人变成了僧侣。他 把人从外在的宗教笃诚解放出来，是因为他把宗教笃诚变成了人的内在世界。他把肉体从锁链中 解放出来，是因为他给人的心灵套上了锁链。 —［德］马克思《〈黑格尔法哲学批判〉导言》，《马克思恩格斯文集》第一卷 史料阅读 法律始终把它的作者的世界图景包含在抽象的形式 中，而每一历史的世界图景都包含一种政治—经济的倾向， 这种倾向依据的不是这个人或那个人所想的事物，却依据 的是事实上掌握政权并因之掌握立法的阶级所实际打算造 成的事物。每一种法律都是由一个阶级以大多数的名义建 立起来的。 —［德］奥斯瓦尔德·斯宾格勒著，齐世荣等译 《西方的没落》 阅读材料，联系课文内容，谈谈你对法律的认识。 探究与拓展 问题探究 英美法系和大陆法系是西方两大法律体系，它们的源 头或多或少地与罗马法有联系，两者之间有共性，也有许 多不同之处。查阅相关资料，进一步了解它们之间的相同 和不同之处。 学习拓展 第10课"

#######################
## 筛选问题
#######################

# f = open("history_questions_candidates2.txt", "r", encoding="utf-8")
# fw = open("history_questions_candidates_filter.txt", 'a', encoding='utf-8')
# l = f.readline()
# while l.strip():
#     data_json = json.loads(l)
#     # print(data_json['text'])
#     # print(data_json['questions'])
#     questions_list = [s[:500] for s in data_json['questions']]
#     inputs = "给定原文：" + data_json['text'][:4000] + \
#              "\n下面是与原文相关的问题列表：" + str(questions_list) + \
#              "\n请从问题列表中选出与原文最相符合的问题。"
#     print(inputs)
#
#     response = openai.ChatCompletion.create(
#         engine="gpt35-34",
#         messages=[
#             {"role": "system",
#              "content": "你现在是一个问题筛选器，需要从问题列表中选出与原文最相关且质量较好的问题。"},
#             {"role": "user", "content": inputs},
#         ],
#         temperature=0.95,
#         max_tokens=800,
#         top_p=0.95,
#         n=5,
#         frequency_penalty=0,
#         presence_penalty=0,
#         stop=None)
#     questions = \
#         [response["choices"][0]["message"]["content"],
#          response["choices"][1]["message"]["content"],
#          response["choices"][2]["message"]["content"],
#          response["choices"][3]["message"]["content"],
#          response["choices"][4]["message"]["content"]]
#     print(questions)
#     json_data = {
#         "text": data_json['text'],
#         "questions": questions
#     }
#     fw.write(json.dumps(json_data, ensure_ascii=False) + '\n')
#     time.sleep(30)
#     l = f.readline()

#######################
## 筛选问题第二步
#######################
# def count_substring_occurrences(strings):
#     result = {}
#     for i, string1 in enumerate(strings):
#         count = 0
#         for j, string2 in enumerate(strings):
#             if string1 in string2:
#                 count += 1
#         result[string1] = count
#     return result
#
#
# f = open("history_questions_candidates_filter.txt", 'r', encoding='utf-8')
# good_num = 0
# l = f.readline()
# while l.strip():
#     data_json = json.loads(l)
#     string_list = data_json['questions']
#     result = count_substring_occurrences(string_list)
#     print(json.dumps(result, ensure_ascii=False))
#
#     good = False
#     for i in result.values():
#         if i >= 2:
#             good = True
#     if good:
#         good_num += 1
#     else:
#         pass
#         # print(data_json['text'])
#
#     l = f.readline()
# print(good_num)

#######################
## 筛选问题第三步
#######################
question_result = []
k_list = []
data_origin = json.load(open("/Users/zhanglin/Desktop/LM_as_Evaluator/data/taxonomy/history.json","r",encoding="utf-8"))
for d in data_origin:
    for k in d.keys():
        for k2 in d[k].keys():
            for k3 in d[k][k2].keys():
                k_list.append(k3)

i = 0
f = open("history_questions_candidates_filter_result.txt", 'r', encoding='utf-8')
l = f.readline()
while l.strip():
    questions_dict = json.loads(l.strip())
    max_count = 0
    best_question = ""
    for k in questions_dict.keys():
        if questions_dict[k] > max_count:
            best_question = k
            max_count = questions_dict[k]
    if max_count < 2:
        best_question = ""
        max_count = 0
    question_result.append({
        "key": k_list[i],
        "best_question": best_question,
        "score": max_count
    })
    l = f.readline()
    i += 1

print(json.dumps(question_result, ensure_ascii=False))