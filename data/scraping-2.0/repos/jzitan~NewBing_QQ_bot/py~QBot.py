import json
import os
import traceback
import uuid
from copy import deepcopy
from flask import request, Flask
import openai
import requests
from text_to_image import text_to_image
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import tiktoken
from text_to_speech import gen_speech
import asyncio
from new_bing import chat_whit_nb, reset_nb_session
from stable_diffusion import get_stable_diffusion_img
from config_file import config_data
from img2prompt import img_to_prompt
import re
import random
from credit_summary import get_credit_summary_by_index

qq_no = config_data['qq_bot']['qq_no']

session_config = {
    'msg': [
        {"role": "system", "content": config_data['chatgpt']['preset']}
    ],
    'send_voice': False,
    'new_bing': True
}

sessions = {}
current_key_index = 0

openai.api_base = "https://chat-gpt.aurorax.cloud/v1"

# 创建一个服务，把当前这个python文件当做一个服务
server = Flask(__name__)


# 测试接口，可以测试本代码是否正常启动
@server.route('/', methods=["GET"])
def index():
    return f"你好，世界!<br/>"


# 获取账号余额接口
@server.route('/credit_summary', methods=["GET"])
def credit_summary():
    return get_credit_summary()


# qq消息上报接口，qq机器人监听到的消息内容将被上报到这里
@server.route('/', methods=["POST"])
def get_message():
    if request.get_json().get('message_type') == 'private':  # 如果是私聊信息
        uid = request.get_json().get('sender').get('user_id')  # 获取信息发送者的 QQ号码
        message = request.get_json().get('raw_message')  # 获取原始信息
        sender = request.get_json().get('sender')  # 消息发送者的资料
        print("收到私聊消息：")
        print(message)
        # 下面你可以执行更多逻辑，这里只演示与ChatGPT对话
        if message.strip().startswith('生成图像'):
            message = str(message).replace('生成图像', '')
            session = get_chat_session('P' + str(uid))
            msg_text = chat(message, session)  # 将消息转发给ChatGPT处理
            # 将ChatGPT的描述转换为图画
            print('开始生成图像')
            pic_path = get_openai_image(msg_text)
            send_private_message_image(uid, pic_path, msg_text)
        elif message.strip().startswith('直接生成图像'):
            message = str(message).replace('直接生成图像', '')
            print('开始直接生成图像')
            pic_path = get_openai_image(message)
            send_private_message_image(uid, pic_path, '')
        elif message.strip().startswith('画图'):
            print("开始stable-diffusion生成")
            pic_url = ""
            try:
                pic_url = sd_img(message.replace("画图", "").strip())
            except Exception as e:
                print("stable-diffusion 接口报错: " + str(e))
                send_private_message(uid, "Stable Diffusion 接口报错: " + str(e), False)
            print("stable-diffusion 生成图像: " + pic_url)
            send_private_message_image(uid, pic_url, '')
        elif message.strip().startswith('下载视频'):
            os.system('rm /root/NewBing/bbdown/video.mp4')
            bv = message.replace("下载视频", "").strip()
            if len(bv)>12:
                bv = bv[bv.find("BV"):bv.find("/?")]
            if len(bv)!=12:
                send_group_message(gid, "请输入正确的网址或BV号", uid, session['send_voice'])
            else:
                print("开始BBDown下载")
                os.system('/root/NewBing/bbdown/BBDown ' + bv + ' --dfn-priority "8K 超高清, 1080P 高码率, HDR 真彩, 杜比视界" --work-dir "/root/NewBing/bbdown" -F "video.mp4" >/dev/null')
                video_url = "/root/NewBing/bbdown/video.mp4"
                print("下载完成")
                send_private_message_video(uid, video_url)
        elif message.strip().startswith('BV'):
            os.system('rm /root/NewBing/bbdown/video.mp4')
            bv = message.strip()
            if len(bv)>12:
                bv = bv[bv.find("BV"):bv.find("/?")]
            if len(bv)!=12:
                send_group_message(gid, "请输入正确的网址或BV号", uid, session['send_voice'])
            else:
                print("开始BBDown下载")
                os.system('/root/NewBing/bbdown/BBDown ' + bv + ' --dfn-priority "8K 超高清, 1080P 高码率, HDR 真彩, 杜比视界" --work-dir "/root/NewBing/bbdown" -F "video.mp4" >/dev/null')
                video_url = "/root/NewBing/bbdown/video.mp4"
                print("下载完成")
                send_private_message_video(uid, video_url)
        elif message.strip().startswith('[CQ:image'):
            print("开始分析图像")
            # 定义正则表达式
            pattern = r'url=([^ ]+)'
            # 使用正则表达式查找匹配的字符串
            match = re.search(pattern, message.strip())
            prompt = img_to_prompt(match.group(1))
            send_private_message(uid, prompt, False)  # 将消息返回的内容发送给用户
        else:
            # 获得对话session
            session = get_chat_session('P' + str(uid))
            if session['new_bing']:
                msg_text = chat_nb(message, session)  # 将消息转发给new bing 处理
            else:
                msg_text = chat(message, session)  # 将消息转发给ChatGPT 
            send_private_message(uid, msg_text, session['send_voice'])  # 将消息返回的内容发送给用户
    if request.get_json().get('message_type') == 'group':  # 如果是群消息
        gid = request.get_json().get('group_id')  # 群号
        uid = request.get_json().get('sender').get('user_id')  # 发言者的qq号
        message = request.get_json().get('raw_message')  # 获取原始信息
        # 判断当被@时才回答
        if str("[CQ:at,qq=%s]" % qq_no) in message:
            sender = request.get_json().get('sender')  # 消息发送者的资料
            print("收到群聊消息：")
            print(message)
            message = str(message).replace(str("[CQ:at,qq=%s]" % qq_no), '')
            if message.strip().startswith('生成图像'):
                message = str(message).replace('生成图像', '')
                session = get_chat_session('G' + str(gid))
                msg_text = chat(message, session)  # 将消息转发给ChatGPT处理
                # 将ChatGPT的描述转换为图画
                print('开始生成图像')
                pic_path = get_openai_image(msg_text)
                send_group_message_image(gid, pic_path, uid, msg_text)
            elif message.strip().startswith('直接生成图像'):
                message = str(message).replace('直接生成图像', '')
                print('开始直接生成图像')
                pic_path = get_openai_image(message)
                send_group_message_image(gid, pic_path, uid, '')
            elif message.strip().startswith('画图'):
                print("开始stable-diffusion生成")
                try:
                    pic_url = sd_img(message.replace("画图", "").strip())
                except Exception as e:
                    print("stable-diffusion 接口报错: " + str(e))
                    send_group_message(gid, "Stable Diffusion 接口报错: " + str(e), uid, False)
                print("stable-diffusion 生成图像: " + pic_url)
                send_group_message_image(gid, pic_url, uid, '')
            elif message.strip().startswith('[CQ:image'):
                print("开始分析图像")
                # 定义正则表达式
                pattern = r'url=([^ ]+)'
                # 使用正则表达式查找匹配的字符串
                match = re.search(pattern, message.strip())
                prompt = img_to_prompt(match.group(1))
                send_group_message(gid, prompt, uid, False)  # 将消息返回的内容发送给用户
            elif message.strip().startswith('下载视频'):
                os.system('rm /root/NewBing/bbdown/video.mp4')
                bv = message.replace("下载视频", "").strip()
                if len(bv)>12:
                    bv = bv[bv.find("BV"):bv.find("/?")]
                if len(bv)!=12:
                    send_group_message(gid, "请输入正确的网址或BV号", uid, session['send_voice'])
                else:
                    print("开始BBDown下载")
                    os.system('/root/NewBing/bbdown/BBDown ' + bv + ' --dfn-priority "8K 超高清, 1080P 高码率, HDR 真彩, 杜比视界" --work-dir "/root/NewBing/bbdown" -F "video.mp4" >/dev/null')
                    video_url = "/root/NewBing/bbdown/video.mp4"
                    print("下载完成")
                    send_group_message_video(gid, video_url, uid)
            elif message.strip().startswith('BV'):
                os.system('rm /root/NewBing/bbdown/video.mp4')
                bv = message.strip()
                if len(bv)>12:
                    bv = bv[bv.find("BV"):bv.find("/?")]
                if len(bv)!=12:
                    send_group_message(gid, "请输入正确的网址或BV号", uid, session['send_voice'])
                else:
                    print("开始BBDown下载")
                    os.system('/root/NewBing/bbdown/BBDown ' + bv + ' --dfn-priority "8K 超高清, 1080P 高码率, HDR 真彩, 杜比视界" --work-dir "/root/NewBing/bbdown" -F "video.mp4" >/dev/null')
                    video_url = "/root/NewBing/bbdown/video.mp4"
                    print("下载完成")
                    send_group_message_video(gid, video_url, uid)
            else:
                # 下面你可以执行更多逻辑，这里只演示与ChatGPT对话
                # 获得对话session
                session = get_chat_session('G' + str(gid))
                if session['new_bing']:
                    msg_text = chat_nb(message, session)  # 将消息转发给new bing处理
                else:
                    msg_text = chat(message, session)  # 将消息转发给ChatGPT处理
                send_group_message(gid, msg_text, uid, session['send_voice'])  # 将消息转发到群里

    if request.get_json().get('post_type') == 'request':  # 收到请求消息
        print("收到请求消息")
        request_type = request.get_json().get('request_type')  # group
        uid = request.get_json().get('user_id')
        flag = request.get_json().get('flag')
        comment = request.get_json().get('comment')
        print("配置文件 auto_confirm:" + str(config_data['qq_bot']['auto_confirm']) + " admin_qq: " + str(
            config_data['qq_bot']['admin_qq']))
        if request_type == "friend":
            print("收到加好友申请")
            print("QQ：", uid)
            print("验证信息", comment)
            # 如果配置文件里auto_confirm为 TRUE，则自动通过
            if config_data['qq_bot']['auto_confirm']:
                set_friend_add_request(flag, "true")
            else:
                if str(uid) == config_data['qq_bot']['admin_qq']:  # 否则只有管理员的好友请求会通过
                    print("管理员加好友请求，通过")
                    set_friend_add_request(flag, "true")
        if request_type == "group":
            print("收到群请求")
            sub_type = request.get_json().get('sub_type')  # 两种，一种的加群(当机器人为管理员的情况下)，一种是邀请入群
            gid = request.get_json().get('group_id')
            if sub_type == "add":
                # 如果机器人是管理员，会收到这种请求，请自行处理
                print("收到加群申请，不进行处理")
            elif sub_type == "invite":
                print("收到邀请入群申请")
                print("群号：", gid)
                # 如果配置文件里auto_confirm为 TRUE，则自动通过
                if config_data['qq_bot']['auto_confirm']:
                    set_group_invite_request(flag, "true")
                else:
                    if str(uid) == config_data['qq_bot']['admin_qq']:  # 否则只有管理员的拉群请求会通过
                        set_group_invite_request(flag, "true")
    return "ok"


# 测试接口，可以用来测试与ChatGPT的交互是否正常，用来排查问题
@server.route('/chat', methods=['post'])
def chatapi():
    requestJson = request.get_data()
    if requestJson is None or requestJson == "" or requestJson == {}:
        resu = {'code': 1, 'msg': '请求内容不能为空'}
        return json.dumps(resu, ensure_ascii=False)
    data = json.loads(requestJson)
    if data.get('id') is None or data['id'] == "":
        resu = {'code': 1, 'msg': '会话id不能为空'}
        return json.dumps(resu, ensure_ascii=False)
    print(data)
    try:
        s = get_chat_session(data['id'])
        msg = chat(data['msg'], s)
        if '查询余额' == data['msg'].strip():
            msg = msg.replace('\n', '<br/>')
        resu = {'code': 0, 'data': msg, 'id': data['id']}
        return json.dumps(resu, ensure_ascii=False)
    except Exception as error:
        print("接口报错")
        resu = {'code': 1, 'msg': '请求异常: ' + str(error)}
        return json.dumps(resu, ensure_ascii=False)


# 重置会话接口
@server.route('/reset_chat', methods=['post'])
def reset_chat():
    requestJson = request.get_data()
    if requestJson is None or requestJson == "" or requestJson == {}:
        resu = {'code': 1, 'msg': '请求内容不能为空'}
        return json.dumps(resu, ensure_ascii=False)
    data = json.loads(requestJson)
    if data['id'] is None or data['id'] == "":
        resu = {'code': 1, 'msg': '会话id不能为空'}
        return json.dumps(resu, ensure_ascii=False)
    # 获得对话session
    session = get_chat_session(data['id'])
    # 清除对话内容但保留人设
    del session['msg'][1:len(session['msg'])]
    resu = {'code': 0, 'msg': '重置成功'}
    return json.dumps(resu, ensure_ascii=False)


# 与new bing交互
def chat_nb(msg, session):
    try:
        if msg.strip() == '':
            return '您好，我是人工智能助手，如果您有任何问题，请随时告诉我，我将尽力回答。\n如果您需要重置我们的会话，请回复`重置会话`'
        if '开启语音' == msg.strip():
            session['send_voice'] = True
            return '语音回复已开启'
        if '关闭语音' == msg.strip():
            session['send_voice'] = False
            return '语音回复已关闭'
        if '重置会话' == msg.strip():
            reset_nb_session(session['id'])
            return '会话已重置'
        if '帮助' == msg.strip():
            return '指令如下(群内需@机器人)：\n1.[模式切换] 请发送 创造、平衡或精确\n2.[重新启动] 请发送 重启\n3.[语音设置] 请发送 开启语音 或 关闭语音\n4.[生成图片] 请发送 画图+描述\n7.[B站视频] 请发送 下载视频+网址/BV号\n6.[指令说明] 请发送 帮助'
        if '重启' == msg.strip():
            os.system("./reboot.sh")
            return "now rebooting"
        if "/gpt" == msg.strip():
            session['new_bing'] = False
            return '已切换至ChatGPT'
        if "创造" ==msg.strip():
            config_data['new_bing']['conversation_style']="h3relaxedimg"
            return '已切换至创造模式'
        if "平衡"==msg.strip():
            config_data['new_bing']['conversation_style']="galileo"
            return '已切换至平衡模式'
        if "精确"==msg.strip():
            config_data['new_bing']['conversation_style']="h3precise"
            return '已切换至精确模式'
        print("问: " + msg)
        replay = asyncio.run(chat_whit_nb(session['id'], msg))
        print("New Bing 返回: " + replay)
        return replay[:-1]
    except Exception as e:
        traceback.print_exc()
        return str('异常: ' + str(e))


# 与ChatGPT交互的方法
def chat(msg, session):
    try:
        if msg.strip() == '':
            return '您好，我是人工智能助手，如果您有任何问题，请随时告诉我，我将尽力回答。\n如果您需要重置我们的会话，请回复`重置会话`'
        if '语音开启' == msg.strip():
            session['send_voice'] = True
            return '语音回复已开启'
        if '语音关闭' == msg.strip():
            session['send_voice'] = False
            return '语音回复已关闭'
        if '重置会话' == msg.strip():
            # 清除对话内容但保留人设
            del session['msg'][1:len(session['msg'])]
            return "会话已重置"
        if '重置人格' == msg.strip():
            # 清空对话内容并恢复预设人设
            session['msg'] = [
                {"role": "system", "content": config_data['chatgpt']['preset']}
            ]
            return '人格已重置'
        if '查询余额' == msg.strip():
            text = ""
            for i in range(len(config_data['openai']['api_key'])):
                text = text + get_credit_summary_by_index(i) + "\n"
            return text
        if '指令说明' == msg.strip():
            return "指令如下(群内需@机器人)：\n1.[重置会话] 请发送 重置会话\n2.[设置人格] 请发送 设置人格+人格描述\n3.[重置人格] 请发送 重置人格\n4.[指令说明] 请发送 " \
                   "指令说明\n注意：\n重置会话不会清空人格,重置人格会重置会话!\n设置人格后人格将一直存在，除非重置人格或重启逻辑端!"
        if msg.strip().startswith('/img'):
            msg = str(msg).replace('/img', '')
            print('开始直接生成图像')
            pic_path = get_openai_image(msg)
            return "![](" + pic_path + ")"
        if msg.strip().startswith('设置人格'):
            # 清空对话并设置人设
            session['msg'] = [
                {"role": "system", "content": msg.strip().replace('设置人格', '')}
            ]
            return '人格设置成功'
        if "/newbing" == msg.strip():
            session['new_bing'] = True
            return '已切换至New Bing'
        # 设置本次对话内容
        session['msg'].append({"role": "user", "content": msg})
        # 设置时间
        session['msg'][1] = {"role": "system", "content": "current time is:" + get_bj_time()}
        # 检查是否超过tokens限制
        while num_tokens_from_messages(session['msg']) > config_data['chatgpt']['max_tokens']:
            # 当超过记忆保存最大量时，清理一条
            del session['msg'][2:3]
        # 与ChatGPT交互获得对话内容
        message = chat_with_gpt(session['msg'])
        # 记录上下文
        session['msg'].append({"role": "assistant", "content": message})
        print("ChatGPT返回内容: ")
        print(message)
        return message
    except Exception as error:
        traceback.print_exc()
        return str('异常: ' + str(error))


# 获取北京时间
def get_bj_time():
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    # 北京时间
    beijing_now = utc_now.astimezone(SHA_TZ)
    fmt = '%Y-%m-%d %H:%M:%S'
    now_fmt = beijing_now.strftime(fmt)
    return now_fmt


# 获取对话session
def get_chat_session(sessionid):
    if sessionid not in sessions:
        config = deepcopy(session_config)
        config['id'] = sessionid
        config['msg'].append({"role": "system", "content": "current time is:" + get_bj_time()})
        sessions[sessionid] = config
    return sessions[sessionid]


def chat_with_gpt(messages):
    global current_key_index
    max_length = len(config_data['openai']['api_key']) - 1
    try:
        if not config_data['openai']['api_key']:
            return "请设置Api Key"
        else:
            if current_key_index > max_length:
                current_key_index = 0
                return "全部Key均已达到速率限制,请等待一分钟后再尝试"
            openai.api_key = config_data['openai']['api_key'][current_key_index]

        resp = openai.ChatCompletion.create(
            model=config_data['chatgpt']['model'],
            messages=messages
        )
        resp = resp['choices'][0]['message']['content']
    except openai.OpenAIError as e:
        if str(e).__contains__("Rate limit reached for default-gpt-3.5-turbo") and current_key_index <= max_length:
            # 切换key
            current_key_index = current_key_index + 1
            print("速率限制，尝试切换key")
            return chat_with_gpt(messages)
        elif str(e).__contains__(
                "Your access was terminated due to violation of our policies") and current_key_index <= max_length:
            print("请及时确认该Key: " + str(openai.api_key) + " 是否正常，若异常，请移除")
            if current_key_index + 1 > max_length:
                return str(e)
            else:
                print("访问被阻止，尝试切换Key")
                # 切换key
                current_key_index = current_key_index + 1
                return chat_with_gpt(messages)
        else:
            print('openai 接口报错: ' + str(e))
            resp = str(e)
    return resp


# 生成图片
def genImg(message):
    img = text_to_image(message)
    filename = str(uuid.uuid1()) + ".png"
    filepath = config_data['qq_bot']['image_path'] + str(os.path.sep) + filename
    img.save(filepath)
    print("图片生成完毕: " + filepath)
    return filename


# 发送私聊消息方法 uid为qq号，message为消息内容
def send_private_message(uid, message, send_voice):
    try:
        if send_voice:  # 如果开启了语音发送
            voice_path = asyncio.run(
                gen_speech(message, config_data['qq_bot']['voice'], config_data['qq_bot']['voice_path']))
            message = "[CQ:record,file=file://" + voice_path + "]"
        if len(message) >= config_data['qq_bot']['max_length'] and not send_voice:  # 如果消息长度超过限制，转成图片发送
            pic_path = genImg(message)
            message = "[CQ:image,file=" + pic_path + "]"
        res = requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/send_private_msg",
                            params={'user_id': int(uid), 'message': message}).json()
        if res["status"] == "ok":
            print("私聊消息发送成功")
        else:
            print(res)
            print("私聊消息发送失败，错误信息：" + str(res['wording']))

    except Exception as error:
        print("私聊消息发送失败")
        print(error)


# 发送私聊消息方法 uid为qq号，pic_path为图片地址
def send_private_message_image(uid, pic_path, msg):
    try:
        message = "[CQ:image,file=" + pic_path + "]"
        if msg != "":
            message = msg + '\n' + message
        res = requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/send_private_msg",
                            params={'user_id': int(uid), 'message': message}).json()
        if res["status"] == "ok":
            print("私聊消息发送成功")
        else:
            print(res)
            print("私聊消息发送失败，错误信息：" + str(res['wording']))

    except Exception as error:
        print("私聊消息发送失败")
        print(error)

# 发送私聊消息方法 uid为qq号，video_path为视频地址
def send_private_message_video(uid, video_path):
    try:
        message = "[CQ:video,file=file://" + video_path + "]"
        res = requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/send_private_msg",
                            params={'user_id': int(uid), 'message': message}).json()
        if res["status"] == "ok":
            print("私聊消息发送成功")
        else:
            print(res)
            print("私聊消息发送失败，错误信息：" + str(res['wording']))

    except Exception as error:
        print("私聊消息发送失败")
        print(error)


# 发送群消息方法
def send_group_message(gid, message, uid, send_voice):
    try:
        if send_voice:  # 如果开启了语音发送
            voice_path = asyncio.run(
                gen_speech(message, config_data['qq_bot']['voice'], config_data['qq_bot']['voice_path']))
            message = "[CQ:record,file=file://" + voice_path + "]"
        if len(message) >= config_data['qq_bot']['max_length'] and not send_voice:  # 如果消息长度超过限制，转成图片发送
            pic_path = genImg(message)
            message = "[CQ:image,file=" + pic_path + "]"
        if not send_voice:
            message = str('[CQ:at,qq=%s]\n' % uid) + message  # @发言人
        res = requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/send_group_msg",
                            params={'group_id': int(gid), 'message': message}).json()
        if res["status"] == "ok":
            print("群消息发送成功")
        else:
            print("群消息发送失败，错误信息：" + str(res['wording']))
    except Exception as error:
        print("群消息发送失败")
        print(error)


# 发送群消息图片方法
def send_group_message_image(gid, pic_path, uid, msg):
    try:
        message = "[CQ:image,file=" + pic_path + "]"
        if msg != "":
            message = msg + '\n' + message
        message = str('[CQ:at,qq=%s]\n' % uid) + message  # @发言人
        res = requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/send_group_msg",
                            params={'group_id': int(gid), 'message': message}).json()
        if res["status"] == "ok":
            print("群消息发送成功")
        else:
            print("群消息发送失败，错误信息：" + str(res['wording']))
    except Exception as error:
        print("群消息发送失败")
        print(error)


# 发送群消息视频方法
def send_group_message_video(gid, video_path, uid):
    try:
        message = "[CQ:video,file=file://" + video_path + "]"
        message = str('[CQ:at,qq=%s]\n' % uid) + message  # @发言人
        res = requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/send_group_msg",
                            params={'group_id': int(gid), 'message': message}).json()
        if res["status"] == "ok":
            print("群消息发送成功")
        else:
            print("群消息发送失败，错误信息：" + str(res['wording']))
    except Exception as error:
        print("群消息发送失败")
        print(error)


# 处理好友请求
def set_friend_add_request(flag, approve):
    try:
        requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/set_friend_add_request",
                      params={'flag': flag, 'approve': approve})
        print("处理好友申请成功")
    except:
        print("处理好友申请失败")


# 处理邀请加群请求
def set_group_invite_request(flag, approve):
    try:
        requests.post(url=config_data['qq_bot']['cqhttp_url'] + "/set_group_add_request",
                      params={'flag': flag, 'sub_type': 'invite', 'approve': approve})
        print("处理群申请成功")
    except:
        print("处理群申请失败")


# openai生成图片
def get_openai_image(des):
    openai.api_key = config_data['openai']['api_key'][current_key_index]
    response = openai.Image.create(
        prompt=des,
        n=1,
        size=config_data['openai']['img_size']
    )
    image_url = response['data'][0]['url']
    print('图像已生成')
    print(image_url)
    return image_url


# 查询账户余额
def get_credit_summary():
    return get_credit_summary_by_index(current_key_index)


# 查询账户余额


# 计算消息使用的tokens数量
def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # 如果name字段存在，role字段会被忽略
                    num_tokens += -1  # role字段是必填项，并且占用1token
        num_tokens += 2
        return num_tokens
    else:
        raise NotImplementedError(f"""当前模型不支持tokens计算: {model}.""")


# sd生成图片,这里只做了正向提示词，其他参数自己加
def sd_img(msg):
    res = get_stable_diffusion_img({
        "prompt": msg,
        "width": 768,
        "height": 512,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "negative_prompt": "",
        "scheduler": "K_EULER_ANCESTRAL",
        "seed": random.randint(1, 9999999),
    }, config_data['replicate']['api_token'])
    return res[0]


if __name__ == '__main__':
    server.run(port=5555, host='0.0.0.0', use_reloader=False)
