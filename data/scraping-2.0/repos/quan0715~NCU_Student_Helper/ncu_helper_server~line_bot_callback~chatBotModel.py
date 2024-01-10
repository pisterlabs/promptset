from line_bot_callback.ncu_wiki import getWikiChainLLM
from .chatBotExtension import chat_status, jump_to, text, button_group, do_nothing, state_ai_agent, quick_reply
from typing import Tuple
from eeclass_setting.models import LineUser
from eeclass_setting.appModel import check_eeclass_update_pipeline, find_user_by_user_id
from django.core.cache import cache
from django.conf import settings
from .views import LineBotCallbackView as cb
from .langChainAgent import LangChainAgent

dog_icon_url = 'https://scontent.xx.fbcdn.net/v/t1.15752-9/387560636_1364784567461116_3708224130470311929_n.png?stp=cp0_dst-png&_nc_cat=108&ccb=1-7&_nc_sid=510075&_nc_ohc=S5L-rQ1y9RcAX9PYIpB&_nc_ad=z-m&_nc_cid=0&_nc_ht=scontent.xx&oh=03_AdQKnpWq0ikfumqNYbstXppxcr2WW48UmGkQgabRCkjJow&oe=655C13D9'

@chat_status("default", default=True)
@quick_reply('選擇指令開啟下一步')
@state_ai_agent(LangChainAgent())
def default_message(event, agent):
    jump_to(main_menu, event.source.user_id)
    return [
        (dog_icon_url, '資料設定', '資料設定'),
        (dog_icon_url, '課程查詢', '課程查詢'),
        (dog_icon_url, '跟我閒聊', agent.run('請給我一個很白癡的問題')),
        (dog_icon_url, 'EECLASS查詢', 'EECLASS查詢'),
        (dog_icon_url, '交通查詢', '交通查詢')
    ]


@chat_status("main menu")
@text
@state_ai_agent(getWikiChainLLM())
# replace LangChainAgent by any agent you want to use
def main_menu(event, aiAgent):
    # aiAgent equal to parameter in @state_ai_agent
    match event.message.text:
        case '資料設定':
            jump_to(default_message, event.source.user_id, propagation=True)
            return settings.FRONT_END_WEB_URL
        case 'EECLASS查詢':
            jump_to(update_eeclass, event.source.user_id, True)
            return
        case '交通查詢':
            jump_to(traffic_message, event.source.user_id, True)
            return
        case 'EECLASS查詢':
            jump_to(ee_query_message, event.source.user_id, True)
        case _:
            jump_to(do_nothing, event.source.user_id)
            try:
                msg = aiAgent(event.message.text)['result']
                jump_to(default_message, event.source.user_id, True)
                # if you update aiAgent and method changed, update code here
                return msg
            except:
                import traceback
                traceback.print_exc()
                jump_to(default_message, event.source.user_id, True)
                return 'error occur by chatbot ai'


@chat_status("traffic message")
@quick_reply("請選擇通勤項目")
def traffic_message(event):
    jump_to(traffic_menu, event.source.user_id)
    return [
        (dog_icon_url, '公車查詢', '公車查詢'),
        (dog_icon_url, '高鐵查詢/訂票', '高鐵查詢/訂票'),
        (dog_icon_url, '返回', '返回')
    ]


@chat_status("traffic_menu")
@text
def traffic_menu(event):
    match event.message.text:
        case '公車查詢':
            jump_to(bus_util, event.source.user_id, False)
            return "請問你想要查什麼公車呢？"
        case '高鐵查詢/訂票':
            jump_to(hsr_util, event.source.user_id, False)
            from backenddb.appModel import find_hsr_data
            hsr_data, founded = find_hsr_data(event.source.user_id)
            warning_msg = "" if founded else "（您還沒設定高鐵訂票所需的個人資訊喔！）"
            return "請問您想要訂哪一天什麼時間的高鐵票呢？" + warning_msg
        case '返回':
            jump_to(default_message, event.source.user_id, True)
            return
        case _:
            return '無此指令'


@chat_status("update eeclass")
@text
def update_eeclass(event):
    search_result:  Tuple[LineUser | None,
                          bool] = find_user_by_user_id(event.source.user_id)
    user, founded = search_result
    if not founded:
        jump_to(default_message, event.source.user_id, True)
        return '尚未設定帳號密碼'
    cb.push_message(event.source.user_id, text(lambda ev: '獲取資料中')(event))
    try:
        result = check_eeclass_update_pipeline(user)
        jump_to(eeclass_util, event.source.user_id, True)
        return result
    except Exception as e:
        jump_to(default_message, event.source.user_id, True)
        return f'獲取失敗, 錯誤訊息:\n{e}'


@chat_status("hsr util")
@text
def hsr_util(event):
    from . import hsrChatbot
    hsr_agent_pool_instance = hsrChatbot.get_agent_pool_instance()
    agent = hsr_agent_pool_instance.get(event.source.user_id)
    if agent is None:
        agent = hsr_agent_pool_instance.add(event.source.user_id)
        agent.run("我要訂高鐵票")
    return agent.run(event.message.text)


@chat_status("eeclass util")
@text
def eeclass_util(event):
    # from . import eeclassChatbot
    # eeclass_agent_pool_instance = eeclassChatbot.get_agent_pool_instance()
    # agent = eeclass_agent_pool_instance.get(event.source.user_id)
    # if agent is None:
    #     agent = eeclass_agent_pool_instance.add(event.source.user_id)
    #     agent.run("我要知道eeclass更新了啥")
    # return agent.run(event.message.text)
    jump_to(default_message, event.source.user_id, True)
    return '請實作eeclassChatBot'

@chat_status("bus util")
@text
def bus_util(event):
    from . import busChatbot
    bus_agent_pool_instance = busChatbot.get_agent_pool_instance()
    agent = bus_agent_pool_instance.get(event.source.user_id)
    if agent is None:
        agent = bus_agent_pool_instance.add(event.source.user_id)
        agent.run("我要查詢公車")
    return agent.run(event.message.text)

@chat_status("ee_query_message")
@button_group("EECLASS查詢項目", "請選擇要查詢的項目", "項目選單")
def ee_query_message(event):
    jump_to(kind_menu, event.source.user_id)
    return [
        '課程',
        '作業',
        '公告',
        '教材'
    ]

@chat_status("kind_menu")
@text
def kind_menu(event):
    match event.message.text:
        case '課程':
            jump_to(about_course, event.source.user_id, False)
            return "請問你想要查什麼哪門課程呢？"
        case '作業' | '公告' | '教材':
            jump_to(about_course, event.source.user_id, False)
            return f"請問您想要查詢關於哪一門課程相關的{event.message.text}呢？"
        case '返回':
            jump_to(default_message, event.source.user_id, True)
            return
        case _:
            return '無此指令'

@chat_status("about_kind")
@text
def about_course(event):
    from . import eeclassBotTool
    ee_agent_pool_instance = eeclassBotTool.get_agent_pool_instance()
    agent = ee_agent_pool_instance.get(event.source.user_id)
    if agent is None:
        agent = ee_agent_pool_instance.add(event.source.user_id)
        agent.run("我要查詢EECLASS資料")
    return agent.run(event.message.text)