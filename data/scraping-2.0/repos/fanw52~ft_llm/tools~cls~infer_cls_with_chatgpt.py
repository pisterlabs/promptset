import time

import jsonlines
import openai

prompt = '''
问题：下面句子包含的事件类型，选项：中毒事故，交通事故，交通逃逸，传销，伤害、殴打他人，伤害致死，其他违法行为，制作传播贩卖淫秽，劫持，劳动生产事故，卖淫嫖娼，危化物品泄漏事故，吸毒，寻衅滋事，建筑坍塌事故，强奸猥亵，意外伤亡，投放危险物质，抢劫抢夺，拐卖，挤压死伤事故，故意伤害，敲诈勒索，斗殴，杀人，求助，涉外事件，涉恐，涉枪，涉邪教，火灾，灾害事故，爆炸爆燃事故，疫情灾害，盗窃，纵火，组织卖淫，绑架，群众求助，自杀，自然天气灾害事故，诈骗，贩毒，赌博，走私，走私毒品，辟谣，阻碍执行职务，非法侵入住宅，非法制售爆炸物品，非法猎捕，非法盗伐林木，非法限制人身自由\n
句子:警情通报8月7日11时01分,杭州市公安局余杭区分局良褚派出所接辖区群众吴女士(化名)报警,称有人偷拍其视频上传网络,并对其恶意中伤。接警后,良褚派出所迅速组织力量开展调查取证工作。经查,网上流传的视频系嫌疑人邮某(男,27岁)趁吴女士在小区快递站点取快递时通过手机摄录。出于博眼球目的,郎某与朋友何某(男,24岁)通过分饰“快递小哥”与“女业主”身份,捏造了暖昧微信聊天内容,并将摄录的吴女士视频和聊天内容截图发至微信群,造成不良社会影响。目前,杭州市公安局佘杭区分局根据《中华人民共和国治安管理处罚法》第四十二条之规定,已对郎某、何某诽谤他人行为分别作出行政拘留处罚\n
结果:上述句子中包含的事件类型包括:其他违法行为\n
句子:#警情通报#关于我市蝴蝶山路某小区“8.6”故意伤害案的情况通报@柳州鱼峰警方在线 #柳州警讯#\n
结果:上述句子中包含的事件类型包括:故意伤害\n
句子:警方通报2018年8月30日晚8时许,我局接群众报警称水碾河公交车站有人打架。民警迅速到达现场控制事态,并将涉事人员杨某(男56岁,本市人)、刘某(男,63岁,本市人)、胡某(男,56岁,本市人)和相关人员带回调查。现查明,杨某在公交车到站准备下车时,与一名女乘客发生肢体碰撞,进而对该女乘客进行辱骂。同车群众滕某误认为杨某有猥亵行为对其进行可序制止,遂遭到杨某等三人殴打。经公安机关调查,杨某无猥亵行为。目前,杨某、刘某、胡某等三人因在公共场所寻衅滋事,于8月31日被依法行政拘留。成都市公安分局锦江区分局2018年9月0平安锦江\n
结果:上述句子中包含的事件类型包括:伤害、殴打他人，寻衅滋事\n
{input_str}\n
结果:
'''

openai.api_key = "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"
result = []
in_dir = "/data/wufan/data/jqtb_cls"
out_dir = "/data/wufan/experiments/llm/chatglm/cls_jqtb"

path = f"{in_dir}/valid.json"
outpath = f"{out_dir}/valid_raw_chatglm.json"

with jsonlines.open(path) as reader:
    for line in reader:
        in_text = line["input"]
        target = line["target"]
        prompt = prompt.format_map({"input_str": in_text})
        print(prompt)
        try:

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            print(response["choices"][0]["message"]["content"])
            line["answer"] = response["choices"][0]["message"]["content"]
        except Exception as e:
            print(e)
            pass
        result.append(line)
        time.sleep(15)

with jsonlines.open(outpath, 'w') as w:
    for line in result:
        w.write(line)
