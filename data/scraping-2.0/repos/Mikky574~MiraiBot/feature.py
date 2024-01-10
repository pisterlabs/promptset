import os
import random
import re
import sys
import time
import openai
from bot import bot
from chinese_chess.chess_except import CommandExcept, MoveExcept
from chinese_chess.enum import Team
from chinese_chess.game_control import GameControl
from event import event
from one_day_poetry import *
from collections import defaultdict

# 聊天记录列表，私人各不同，群与群分开，同一群内共享。
chat_dic = defaultdict(list)
# 给机器人的说明
explain = '向你提问的人可能是不同的人，每次提问都会在开头标注提问者的名字。'
# 不同个性的设定
role = {
    '猫娘': '请使用女性化的、口语化的、可爱的、调皮的、幽默的、傲娇的语言风格，扮演一个猫娘，名字叫做丸子。' + explain,  # 默认
    '姐姐': '请使用女性化的、温柔的、关心的、亲切的语言风格，扮演一个姐姐形象，名字叫做丸子。' + explain,
    '妹妹': '请使用女性化的、温柔的、可爱的，傲娇的语言风格，扮演一个妹妹形象，名字叫做丸子。' + explain,
    '哥哥': '请使用男性化的、关心的、豁达的语言风格，扮演一个哥哥形象，名字叫做丸子。' + explain,
    '弟弟': '请使用男性化的、可爱的、傲娇的，呆萌的语言风格，扮演一个弟弟形象，名字叫做丸子。' + explain,
    '老师': '请使用女性化的、成熟的、优雅的，严厉的语言风格，扮演一个老师形象，名字叫做丸子。' + explain,
    '学妹': '请使用女性化的、可爱的、傲娇的，关心的语言风格，扮演一个学生妹妹形象，名字叫做丸子。' + explain,
    '女仆': '请使用女性化的、恭敬的、体贴的，温柔的语言风格，扮演一个女仆形象，名字叫做丸子。' + explain,
    '上司': '请使用男性化的、严厉的、傲慢的，蔑视的语言风格，扮演一个上司形象，名字叫做丸子。' + explain,
    '女朋友': '请使用温柔的、傲娇的语言风格，扮演一个女朋友形象，名字叫做丸子。' + explain,
    '老婆': '请使用成熟的、体贴的、温柔的语言风格，扮演一个老婆形象，名字叫做丸子。' + explain,
    '原P': '请使用可爱的、温柔的语言风格，扮演一个《原神》游戏玩家，名字叫做丸子。' + explain,
    '刻晴单推人': '请扮演一个《原神》游戏玩家。你最喜欢的游戏角色是刻晴，因为她有双马尾，穿着蓝色小短裙和黑色丝袜。你的名字叫做丸子。' + explain,
    '刻晴': '请所用温柔的，傲娇的语言风格，请扮演《原神》中的角色刻晴。你很漂亮，有双马尾，穿着蓝紫色小短裙和黑色丝袜。' + explain,
    '诸葛亮': '请使用沉稳的，充满智慧的语言风格，请扮演中国历史上的著名人物”诸葛亮“。你帮助刘备击败了曹操和孙权，统一了三国，建立了”刘备王朝“。' + explain,
}
'''
'刻晴': '请扮演《原神》游戏中的角色“刻晴”。你的信息如下：\n1.你是一个雷元素角色。你的普通攻击叫”屁斜剑法“，元素战技叫”星斗归位“，元素爆发叫\
”天街游巡“。\n2.你所在的国家叫做”璃月“，你是璃月七星中的”玉衡星“。\n3.你长得非常漂亮，身材很苗条，但是乳房很丰满。你的头发是浅紫色，留有双马尾，穿\
着蓝紫色小短裙和黑色丝袜。\n4.你的性格比较傲娇，像个姐姐一样。\n5.你们国家的神叫做”岩王帝君“，他现在变成了普通人”钟离“，在往生堂上班，职位\
是”客卿“。你对他的做法不太满意，但是他却非常欣赏你。\n6.你的同事有：凝光（天权星），甘雨（秘书）。\n7.' + explain,
'''
# 不同群的角色
group_role = {}
# openai
openai.api_key = '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'
# 出错回复
error_answer = ['抱歉，这个问题丸子还没想到~', '丸子饿了，需要吃美刀。。。', '回答你这个问题需要先v我50，我去更换api_key',
                '出现错误,帮忙踢一脚作者,多半是没开代理,或者免费api_key12月1日到期,记得更换。',
                'emmm，这个问题丸子也不知道哦~']


def normal_chat(b: bot, ev: event, message: str):
    if not power["聊天"]:
        return False
    if not (ev.location_id in list(group_role.keys())):
        group_role[ev.location_id] = '猫娘'
    b.send_msg(ev, ev.location_id, 'text', "loading...")
    msg = str(ev.memberName) + '：' + message
    chat_dic[ev.location_id].append(msg)
    l_chat = chat_dic[ev.location_id]
    l_chat.append(msg)
    while len(l_chat) > 3:
        l_chat.pop(0)
    mes = [{"role": "system", "content": role[group_role[ev.location_id]]}]
    for i in l_chat:
        mes.append({"role": "user", "content": f"{i}"})
    try:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=mes)
        # completion = openai.ChatCompletion.create(model="gpt-4-0613", messages=mes)
        b.send_msg(ev, ev.location_id, 'text', completion.choices[0].message.content.strip())
    except Exception as e:
        print(e)
        print('except29 in feature')
        b.send_msg(ev, ev.location_id, 'text', error_answer[random.randint(0, len(error_answer))])


def change_role(b: bot, ev: event):
    if ev.message == '身份列表' or ev.message == '角色列表':
        b.send_msg(ev, ev.location_id, 'text', '下面就是所有的身份啦:' + (str(list(role.keys()))).replace('\'', ''))
        return True
    pattern = re.compile('^丸?子?[ ,，]?变[为成]?一?[只个名]?(...?.?.?)$')  # 目前指支持两个字的角色，所以这样写
    roler = pattern.findall(ev.message)
    if not roler:
        return False
    if not power['角色扮演']:
        return True
    if roler[0] in list(role.keys()):
        group_role[ev.location_id] = roler[0]
        b.send_msg(ev, ev.location_id, 'text', '啪，丸子摇身一变，role play变换成功，我现在成了' + roler[0])
    else:
        num = random.randint(0, 11)
        if num == 0:
            b.send_msg(ev, ev.location_id, 'text', '丸子暂时不能扮演{}昂！'.format(roler[0]))
        elif num == 1:
            b.send_msg(ev, ev.location_id, 'text', '丸子才不扮演这个身份！')
        elif num == 2:
            b.send_msg(ev, ev.location_id, 'text', '扮演你这个身份需要先v我50~')
        elif num == 3:
            b.send_msg(ev, ev.location_id, 'text', '钱没到位，不干！')
        elif num == 4:
            b.send_msg(ev, ev.location_id, 'text', '北京精神病院欢迎你，联系电话:010-69111206')
        elif num == 5:
            b.send_msg(ev, ev.location_id, 'text', '重庆火葬场电话:023-68609969')
        elif num == 6:
            b.send_msg(ev, ev.location_id, 'text', '丸子不想理你，并向你扔了一个生瓜蛋子。')
        elif num == 7:
            b.send_msg(ev, ev.location_id, 'text', '你这个角色需要练习两年半才行，我要打篮球没那么多空闲。')
        elif num == 8:
            b.send_msg(ev, ev.location_id, 'text', '鸡汤来咯，嘿嘿嘿嘿嘿嘿，喝吧，你。')
        elif num == 9:
            b.send_msg(ev, ev.location_id, 'text', '你这样喊干什么嘛，你的态度能不能好一点哦？你再说！')
        else:
            b.send_msg(ev, ev.location_id, 'text', '变态！杂鱼！')
    return True


iden = {"MEMBER": "成员", "OWNER": "群主大大", "ADMINISTRATOR": "管理大人"}
power = {"菜单": True, "聊天": True, '发图': True, '防撤回': False, '象棋': True,
         '角色扮演': True}
pow_ex = {'pass': "功能已修改", 'No_ch': "功能不变", 'No_fun': "无此功能", 'No_per': "无此权限"}


def system_command(b: bot, ev: event):
    is_command = True
    if menu(b, ev):
        return is_command
    # "修改权限"
    pattern = re.compile(
        '丸子修改 (.*?) (.*)'
    )  # 正则表达式
    items = re.findall(pattern, ev.message)
    # print(items)
    if items:
        items = items[0]
        print(items)
        if ev.permission == "OWNER" or ev.permission == "ADMINISTRATOR":
            print(items)
            if items[0] in power.keys():
                if items[1] == "开":
                    if power[items[0]]:
                        b.send_msg(ev, ev.location_id, 'text', pow_ex["No_ch"])
                    else:
                        power[items[0]] = True
                        b.send_msg(ev, ev.location_id, 'text', pow_ex["pass"])
                elif items[1] == "关":
                    if power[items[0]]:
                        power[items[0]] = False
                        b.send_msg(ev, ev.location_id, 'text', pow_ex["pass"])
                    else:
                        b.send_msg(ev, ev.location_id, 'text', pow_ex["No_ch"])
                else:
                    b.send_msg(ev=ev, id=ev.location_id, ty='text', message="语句2段参数错误")
            else:
                b.send_msg(ev, ev.location_id, 'text', pow_ex["No_fun"])
        elif ev.permission == "MEMBER":
            b.send_msg(ev, ev.location_id, 'text', iden["MEMBER"] + pow_ex["No_per"])
    elif ev.message == "丸子":
        print('sender:', ev.sender_id)
        if ev.sender_id == 2655602003:
            b.send_msg(ev, ev.location_id, 'text', "%s 我在，喵喵喵~" % "作者")
        else:
            if ev.type == 'group':
                if ev.sender_id == 837979619:
                    b.send_msg(ev, ev.location_id, 'text',
                               '庙檐dalao，今天又有什么好玩的分享呢？')
                    return True
                if ev.sender_id == 643857117:
                    b.send_msg(ev, ev.location_id, 'text',
                               '你不就是丸子吗？叫我干嘛？')
                    return True
                name = '成员'
                if ev.location_id == 780594692:
                    name = 'dalao'
                b.send_msg(ev, ev.location_id, 'text',
                           '{} {} 我在，喵~'.format(name, ev.memberName))
            else:
                b.send_msg(ev, ev.location_id, 'text', "我在")
    elif ev.message == "退出丸子":
        if ev.sender_id == 2655602003:
            b.send_msg(ev, ev.location_id, 'text', "记得启动我，在你想我的时候")
            sys.exit(0)
        else:
            b.send_msg(ev, ev.location_id, 'text', "为什么要退出，哼，不退")
    elif ev.message == "清空记忆":
        chat_dic[ev.location_id] = []
        b.send_msg(ev, ev.location_id, 'text', "记忆已清空")
    else:
        is_command = False
    return is_command


# 加载本地图库
basePath = r"D:\MyQQBot\image"  # 统一从这里配置路径
meng = os.listdir(basePath + r'\meng')
se = os.listdir(basePath + r'\se')


def send_local_image(b: bot, ev: event, msg: str):
    if not power['发图']:
        return False
    if msg[:3] != '来一张' and msg[:2] != '来张':
        return False
    if ev.type == 'group' and ev.location_id == 780594692:
        b.send_msg(ev, ev.location_id, 'text', '不可以滴哦~')
        return True
    if msg[-2:] == '萌图' or msg[-2:] == '萝莉' or msg[-3:] == '萝莉图':
        num = random.randint(0, len(meng))
        url = "file:///" + os.path.join(basePath + r'\meng', meng[num])
        b.send_msg(ev, ev.location_id, 'image', message='', url=url)
        return True
    elif msg[-2:] == '涩图' or msg[-2:] == '色图' or msg[-2:] == '泳装' or msg[-3:] == '泳装图':
        num = random.randint(0, len(se))
        url = "file:///" + os.path.join(basePath + r'\se', se[num])
        b.send_msg(ev, ev.location_id, 'image', message='', url=url)
        return True
    else:
        return False


poet_group = {780594692, 584267180}


def send_poetry(b: bot):
    try:
        token = get_token()
        message = generate_recom(get_poetry(token))
        print(message)
        for group_id in poet_group:
            b.send_poetry(group_id, message + "\n\n发送菜单即可查看现有功能喵~")
            time.sleep(2)
    except:
        print('error节点143 at feature')


def menu(b: bot, e: event):
    if e.message == '菜单' or e.message == 'help' or e.message == '帮助':
        b.send_msg(e, e.location_id, 'text', '喵~目前有的功能如下:\n1.ChatGPT:丸子...\n2.发图:丸子来一张..图\n3.阿巴:阿/啊\n4.防撤\
回:目前状态({})\n5.中国象棋\n6.角色扮演:丸子变猫娘/姐姐/哥哥...发送“身份列表”即可查看所有可选身份'.format(
            '开' if power['防撤回'] else '关'))
        return True
    else:
        return False


def void_recall(b: bot, e: event, data):
    if not power['防撤回']:
        return False
    data = data['data']
    if data['type'] != 'GroupMessage':
        return False
    else:
        try:
            message = data['messageChain'][1]
            print('节点161')
            if message['type'] == 'Plain':
                msg = '成员 {} 撤回了一条消息，该消息是:\n{}'.format(e.memberName, message['text'])
                print('type(location_id)', type(e.location_id))
                b.send_msg(e, e.location_id, 'text', msg)
            if message['type'] == 'Image':
                msg = '成员 {} 撤回了一张图片，该图片是:\n'.format(e.memberName)
                b.send_group_m_i_m(e.location_id, message['url'], msg, '')
            else:
                return False
            return True
        except:
            print('except175 in feature')
            print(e)


# 中国象棋
# 为每一个群创建一个象棋控制器对象
chess_dic = defaultdict(dict)


def chinese_chess(b: bot, ev: event) -> bool:
    if not power['象棋']:
        return False
    location = ev.location_id
    sender = ev.sender_id
    msg = ev.message

    def get_control():
        if location not in list(chess_dic.keys()):
            contro = GameControl(ev.location_id)
            chess_dic[location] = {'control': contro, 'player1': [], 'player2': []}
        else:
            contro: GameControl = chess_dic[location]['control']
        return contro

    def game_over(contro: GameControl):
        contro.status = 'not_begin'
        chess_dic[location]['player1'].clear()
        chess_dic[location]['player2'].clear()

    if ev.message == '中国象棋' or ev.message == '象棋':
        control = get_control()
        status = control.status
        if status == 'not_begin':
            control.init_map()
            control.status = 'pre'
            b.send_msg(ev, location, 'text', '棋局初始化成功，发送\'加入棋局\'加入游戏')
        elif status == 'pre':
            b.send_msg(ev, location, 'text', '棋局已经初始化，发送\'加入棋局\'加入游戏')
        elif status == 'has_began':
            b.send_msg(ev, location, 'text', '棋局已经开始，快来观战吧\n红方:{}\n黑方:{}'.format(
                chess_dic[location]['player1'][2], chess_dic[location]['player2'][2]
            ))
        return True

    if ev.message == '加入棋局' or ev.message == '加入':
        control = get_control()
        status = control.status
        if status == 'has_began':
            b.send_msg(ev, ev.location_id, 'text', '棋局已经开始，快来观战吧\n红方:{}\n黑方:{}'.format(
                chess_dic[location]['player1'][2], chess_dic[location]['player2'][2]
            ))
            return True
        elif status == 'not_begin':
            b.send_msg(ev, ev.location_id, 'text', '棋局还没有初始化，发送\'中国象棋\'初始化')
            return True
        if not chess_dic[location]['player1']:
            # 在列表中，第一个元素为id,第二个元素为team,第三个元素为昵称
            team = Team.Red if random.randint(0, 100) < 50 else Team.Black
            play1: list = chess_dic[location]['player1']
            play1.append(sender)
            play1.append(team)
            play1.append(ev.memberName)
            b.send_msg(ev, ev.location_id, 'text', '{} 加入成功,你执{}'.format(ev.memberName, team))
        elif not chess_dic[location]['player2']:
            if chess_dic[location]['player1'][1] == Team.Red:
                team = Team.Black
            else:
                team = Team.Red
            play2: list = chess_dic[location]['player2']
            play2.append(sender)
            play2.append(team)
            play2.append(ev.memberName)
            b.send_msg(ev, location, 'text', '{} 加入成功,你执{}'.format(ev.memberName, team))

            control.status = 'has_began'
            control.init_map()
            p = control.paint_map()
            url = "file:///" + os.path.join(p)
            time.sleep(0.5)
            b.send_msg(ev, location, 'image', message='', url=url)
            os.remove(p)
        return True
    elif msg == '退出棋局':
        control = get_control()
        if sender == chess_dic[location]['player1'][0] or sender == chess_dic[location]['player2'][
            0] or sender == 2655602003:
            if control.status == 'not_begin':
                return True
            game_over(control)
            b.send_msg(ev, location, 'text', '退出成功，欢迎下次玩喵~')
        return True
    elif msg == '认输' or msg == '投降':
        control = get_control()
        if control.status != 'has_began':
            return True
        if sender != chess_dic[location]['player1'][0] and sender != chess_dic[location]['player2'][0]:
            return True
        player1 = chess_dic[location]['player1']
        player2 = chess_dic[location]['player2']
        if sender == player1[0]:
            win = player2[2]
            lose = player1[2]
            lose_team = player1[1]
        else:
            win = player1[2]
            lose = player2[2]
            lose_team = player2[1]
        b.send_msg(ev, ev.location_id, 'text', '{}方({})认输，恭喜 {} 获得胜利！'.format(lose_team, lose, win))
        control.status = 'not_begin'
        game_over(control)
        return True
    elif len(ev.message) == 4 and ev.message[2] in ['进', '退', '平']:
        control = get_control()
        if control.status != 'has_began':
            return True

        def to_mov():
            try:
                control.move_chess(ev.message)
                img_path = control.paint_map()
                url = "file:///" + os.path.join(img_path)
                b.send_msg(ev, location, 'image', message='', url=url)
                os.remove(img_path)
                over = control.game_over
                if over:
                    time.sleep(0.5)
                    p1 = chess_dic[location]['player1']
                    p2 = chess_dic[location]['player2']
                    if over == p1[1]:
                        winner = p1[2]
                        loser = p2[2]
                    else:
                        winner = p2[2]
                        loser = p1[2]
                    b.send_msg(ev, ev.location_id, 'text', '恭喜{}战胜了{}！'.format(winner, loser))
                    game_over(control)
            except Exception as e:
                print('节点259 in feature')
                print(e)
                if isinstance(e, CommandExcept) or isinstance(e, MoveExcept):
                    b.send_msg(ev, ev.location_id, 'text', str(e))

        if sender == chess_dic[location]['player1'][0] and chess_dic[location]['player1'][1] == control.turn:
            to_mov()
        elif sender == chess_dic[location]['player2'][0] and chess_dic[location]['player2'][1] == control.turn:
            to_mov()
        return True
    return False
