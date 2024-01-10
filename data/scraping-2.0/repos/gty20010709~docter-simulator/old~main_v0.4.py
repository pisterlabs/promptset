# -- 2023/4/30 -- #

# ////////////////////////////////////////////////////////////////////
# //                          _ooOoo_                               //
# //                         o8888888o                              //
# //                         88" . "88                              //
# //                         (| ^_^ |)                              //
# //                         O\  =  /O                              //
# //                      ____/`---'\____                           //
# //                    .'  \\|     |//  `.                         //
# //                   /  \\|||  :  |||//  \                        //
# //                  /  _||||| -:- |||||-  \                       //
# //                  |   | \\\  -  /// |   |                       //
# //                  | \_|  ''\---/''  |   |                       //
# //                  \  .-\__  `-`  ___/-. /                       //
# //                ___`. .'  /--.--\  `. . ___                     //
# //              ."" '<  `.___\_<|>_/___.'  >'"".                  //
# //            | | :  `- \`.;`\ _ /`;.`/ - ` : | |                 //
# //            \  \ `-.   \_ __\ /__ _/   .-` /  /                 //
# //      ========`-.____`-.___\_____/___.-`____.-'========         //
# //                           `=---='                              //
# //      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^        //
# //            佛祖保佑       永不宕机     永无BUG                    //
# ////////////////////////////////////////////////////////////////////

import asyncio
import aiofiles
import flet as ft
import webbrowser
from enum import Enum
from info import home_message, notification
from bot import prompts, request_para
import openai
from dotenv import load_dotenv
import time

load_dotenv()

# ---------------------- 全局配置，以便便捷地修改界面和功能 ------------
# 通过增删此列表，可迅速改变侧边栏上目标的个数
NAV_RAIL_DESTINATIONS: list[str] = ["P1", "P2", "P3"]
# 按此比例配置聊天消息宽度
WIDTH_PERCENT = 0.7

BOT_NAME = "Patient"
HUMAN_NAME = "Doctor"

TURNS = 5  # 携带多少轮上下文


class Sender(Enum):
    HUMAN = "human"
    BOT = "bot"


class Bot:
    async def get_respond(
        human_message_text,
        human_history_message_list,
        bot_history_message_list,
        nav_rail_selected_index,
    ):
        l1 = [{"role": "user", "content": x} for x in human_history_message_list]
        l2 = [{"role": "assistant", "content": x} for x in bot_history_message_list]
        exist_message_ls = [x for pair in zip(l1, l2) for x in pair]
        messages = (
            [{"role": "system", "content": prompts[nav_rail_selected_index]}]
            + exist_message_ls
            + [{"role": "user", "content": human_message_text}]
        )

        try:
            result = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo", messages=messages, stream=False, **request_para
            )
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return str(e)


async def main(page: ft.Page):
    # 测试时清除数据
    # await page.client_storage.clear_async()

    ################################################
    # ----------- 初始化客户端数据配置 -------------- #
    ###############################################
    async def if_not_set_default(key, value):
        if not await page.client_storage.contains_key_async(key):
            await page.client_storage.set_async(key, value)

    await if_not_set_default("theme_mode", "light")  # 默认亮色主题
    await if_not_set_default("new_vistor", True)  # 默认第一次访问
    await if_not_set_default("nav_rail_visible", True)
    await if_not_set_default("nav_rail_selected_index", "None")  # 初始界面
    await if_not_set_default("lastest_view", "nav_rail_leading")  # 记住最新加载的界面

    # 初始根据 destination 的数量初始化聊天记录列表
    # 单条聊天记录格式 ["hello","human"] ["hello, how can i assistant you","bot"]
    await if_not_set_default("records", [[] for x in range(len(NAV_RAIL_DESTINATIONS))])
    try:
        # 如果后来增加了 destination 可以在初始的基础上增加
        records = await page.client_storage.get_async("records")
        if len(records) < len(NAV_RAIL_DESTINATIONS):
            number = len(NAV_RAIL_DESTINATIONS) - len(records)
            records += [[] for x in range(number)]
            await page.client_storage.set_async("records", records)
    except Exception as e:
        print(e)

    #################################################
    # -------------- 可共用控件及函数 --------------- #
    ################################################

    async def get_client_info(e):
        try:
            client_ip = page.client_ip
        except:
            client_ip = None
        try:
            client_user_agent = page.client_user_agent
        except:
            client_user_agent = None

        client_platform = page.platform

        return client_ip, client_platform, client_user_agent

    async def on_keyboard(e: ft.KeyboardEvent):
        if new_message_textfield.value == "" and e.key == "Tab":
            new_message_textfield.value = "您好，请问您怎么称呼？我是医生"
            await new_message_textfield.update_async()

        if (
            e.ctrl
            and e.key == "L"
            and await page.client_storage.get_async("lastest_view")
            == "nav_rail_destinations"
        ):
            await clean_chat_records(None)

    async def load_history_records_to_chat_listview():
        records = await page.client_storage.get_async("records")
        for item in records[nav_rail.selected_index]:
            await build_message_row(
                *item,
                (page.width if page.width > 0 else page.window_width),
                stream=False,
            )

        await chat_listview.update_async()

    async def add_notification_to_chatlistview():
        chat_listview.controls = [
            ft.Markdown(
                value=notification,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                on_tap_link=lambda e: webbrowser.open(e.data),
            )
        ]
        chat_listview.auto_scroll = False
        await chat_listview.update_async()
        chat_listview.auto_scroll = True

    async def change_theme(e):
        # print(page.theme_mode)
        page.theme_mode = "light" if page.theme_mode == "dark" else "dark"
        await page.update_async()
        await page.client_storage.set_async("theme_mode", page.theme_mode)

    async def clean_chat_records(e):
        chat_listview.controls = []
        await add_notification_to_chatlistview()
        records = await page.client_storage.get_async("records")
        records[nav_rail.selected_index] = []
        await page.client_storage.set_async("records", records)
        # await chat_listview.update_async()

    async def get_lastest_n_records(turns: TURNS):
        records = await page.client_storage.get_async("records")
        the_records = records[nav_rail.selected_index]
        if len(the_records) < turns:
            # 消息小于5轮，返回全部
            human_history_messages = [x[0] for x in the_records if x[1] == "human"]
            bot_history_messages = [x[0] for x in the_records if x[1] == "bot"]
        else:
            the_records = the_records[: turns * 2]
            human_history_messages = [x[0] for x in the_records if x[1] == "human"]
            bot_history_messages = [x[0] for x in the_records if x[1] == "bot"]

        return (human_history_messages, bot_history_messages)

    ####################################################
    # -------------------- 页面属性 ------------------- #
    ###################################################
    page.title = "Doctor Simulator"
    try:
        page.theme_mode = await page.client_storage.get_async("theme_mode")
    except Exception as e:
        print(f"{page.platform}在加载主题时出错:{e}")
    page.on_keyboard_event = on_keyboard

    #############################################################
    # -------------- 侧边栏相关的控件和函数 ----------------------- #
    #############################################################
    async def nav_rail_leading_on_click(e):
        await page.client_storage.set_async("lastest_view", "nav_rail_leading")
        bottom_row.visible = False
        clean_chat_records_btn_in_appbar.visible = False
        clean_chat_records_btn_in_bottom_bar.visible = False
        nav_rail.selected_index = None
        await page.client_storage.set_async("nav_rail_selected_index", "None")

        chat_listview.controls = [
            ft.Column(
                [
                    ft.Markdown(
                        value=home_message,
                        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                        on_tap_link=lambda e: webbrowser.open(e.data),
                    ),
                ],
            )
        ]
        chat_listview.auto_scroll = False
        await page.update_async()
        chat_listview.auto_scroll = True

    async def nav_rail_destinations_on_click(e):
        if page.width < page.height:
            clean_chat_records_btn_in_appbar.visible = True
            await page.appbar.update_async()
        elif page.width > page.height:
            clean_chat_records_btn_in_bottom_bar.visible = True

        nav_rail_switch.icon = (
            ft.icons.KEYBOARD_ARROW_LEFT
            if nav_rail.visible
            else ft.icons.KEYBOARD_ARROW_RIGHT
        )
        bottom_row.visible = True
        await bottom_row.update_async()

        chat_listview.controls = []
        await load_history_records_to_chat_listview()
        if chat_listview.controls == []:
            await add_notification_to_chatlistview()

        await page.client_storage.set_async("lastest_view", "nav_rail_destinations")
        await page.client_storage.set_async(
            "nav_rail_selected_index", nav_rail.selected_index
        )

    async def nav_rail_trailing_on_click(e):
        await page.client_storage.set_async("lastest_view", "nav_rail_trailing")
        bottom_row.visible = False
        clean_chat_records_btn_in_appbar.visible = False
        clean_chat_records_btn_in_bottom_bar.visible = False
        nav_rail.selected_index = None
        await page.client_storage.set_async("nav_rail_selected_index", "None")

        chat_listview.controls = []

        chat_listview.controls.append(statement)
        chat_listview.controls.append(feedback_card)

        chat_listview.auto_scroll = False
        await page.update_async()
        chat_listview.auto_scroll = True

    async def feedback_btn_on_click(e):
        client_ip, client_platform, client_user_agent = await get_client_info(None)
        feedback_content = feedback_textfield.value
        if feedback_content != "":
            async with aiofiles.open("feedback.log", "a", encoding='utf8') as fi:
                await fi.write(
                    f"""
    Time:{time.ctime()}
    IP:{client_ip}
    Platform:{client_platform}
    User-Agent:{client_user_agent}
    Feedback:{feedback_content}
        """
                )
            feedback_textfield.value = ""
            await feedback_card.update_async()

            page.snack_bar = ft.SnackBar(ft.Text("反馈成功，感谢您的宝贵意见~~"))
            page.snack_bar.open = True
            await page.update_async()

    statement = ft.Card(
        content=ft.ListTile(
            leading=ft.Icon(ft.icons.ALBUM),
            title=ft.Text("免责声明", color="red", size=18),
            subtitle=ft.Text("本站病人纯属虚构，与现实中的人物无任何联系，请在使用过程中遵守法律法规。"),
        )
    )

    feedback_textfield = ft.TextField(
        label="我觉得网站有可以改进的地方~",
        multiline=True,
        max_lines=10,
        min_lines=5,
    )
    feedback_btn = ft.ElevatedButton(
        icon=ft.icons.SEND_ROUNDED, text="提交反馈", on_click=feedback_btn_on_click
    )

    feedback_card = ft.Card(
        content=ft.ListTile(
            leading=ft.Icon(ft.icons.ALBUM),
            title=ft.Text("反馈", size=18),
            subtitle=ft.Column(
                [
                    feedback_textfield,
                    ft.Row([feedback_btn], alignment=ft.MainAxisAlignment.END),
                ],
                alignment=ft.CrossAxisAlignment.STRETCH,
            ),
        )
    )

    nav_rail = ft.NavigationRail(
        leading=ft.IconButton(
            icon=ft.icons.HOUSE, tooltip="Home界面", on_click=nav_rail_leading_on_click
        ),
        group_alignment=-0.5,
        destinations=[
            ft.NavigationRailDestination(icon=ft.icons.MAN, label=x)
            for x in NAV_RAIL_DESTINATIONS
        ],
        expand=1,
        on_change=nav_rail_destinations_on_click,
        trailing=ft.IconButton(
            icon=ft.icons.SETTINGS_ROUNDED,
            tooltip="设置",
            on_click=nav_rail_trailing_on_click,
        ),
        visible=await page.client_storage.get_async("nav_rail_visible"),
        selected_index=(
            None
            if await page.client_storage.get_async("nav_rail_selected_index") == "None"
            else await page.client_storage.get_async("nav_rail_selected_index")
        ),
    )

    #####################################################
    # ------------- AppBar 相关的控件和函数 ------------- #
    ####################################################
    clean_chat_records_btn_in_appbar = ft.IconButton(
        icon=ft.icons.DELETE,
        tooltip="清除聊天记录",
        on_click=clean_chat_records,
        visible=False,
    )

    page.appbar = ft.AppBar(
        title=ft.Row(
            [
                ft.Icon(ft.icons.LOCAL_HOSPITAL_OUTLINED),
                ft.Text("Doctor Simulator"),
            ]
        ),
        actions=[
            ft.IconButton(ft.icons.LIGHTBULB, on_click=change_theme, tooltip="点击改变主题"),
            clean_chat_records_btn_in_appbar,
        ],
        toolbar_height=40,
    )

    #####################################################################
    # ----------------- chat list view 相关控件和函数 ------------------- #
    #####################################################################

    async def build_message_row(message_text: str, sender, page_width, stream=False):
        async def get_initials(name: str):
            return name[:1].capitalize()

        # 此处不能直接用 message_markdown_view = markdown_view
        # 如果，两个指向同一个对象，并不独立
        message_markdown_view = ft.Markdown(
            value="",
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            on_tap_link=lambda e: webbrowser.open(e.data),
        )
        message_row = ft.Row(
            [
                ft.CircleAvatar(
                    content=ft.Text(
                        await get_initials(
                            BOT_NAME if sender == Sender.BOT.value else HUMAN_NAME
                        )
                    ),
                    color=ft.colors.WHITE,
                    bgcolor=(
                        ft.colors.PURPLE
                        if sender == Sender.BOT.value
                        else ft.colors.BROWN
                    ),
                ),
                ft.Column(
                    [
                        message_markdown_view,
                    ],
                    width=int(page_width * WIDTH_PERCENT),
                    tight=True,
                    spacing=5,
                    alignment="start",
                ),
            ],
            vertical_alignment="center",
            alignment="start",
        )

        chat_listview.controls.append(message_row)

        await chat_listview.update_async()
        # print(chat_listview.controls)
        if sender == Sender.HUMAN.value or stream == False:
            message_markdown_view.value = message_text
            await chat_listview.update_async()
        elif sender == Sender.BOT.value:
            for x in message_text:
                message_markdown_view.value += x
                await asyncio.sleep(0.04)
                await message_markdown_view.update_async()
            await chat_listview.update_async()

    chat_listview = ft.ListView(expand=True, spacing=10, auto_scroll=True)

    ######################################################################
    # ------------------ bottom row 相关控件和设置 ----------------------- #
    #####################################################################

    async def nav_rail_switch_on_click(e):
        nav_rail.visible = False if nav_rail.visible else True
        nav_rail_switch.icon = (
            ft.icons.KEYBOARD_ARROW_LEFT
            if nav_rail.visible
            else ft.icons.KEYBOARD_ARROW_RIGHT
        )
        await bottom_row.update_async()
        await page.client_storage.set_async("nav_rail_visible", nav_rail.visible)
        await nav_rail.update_async()

    async def send_message_btn_on_click(e):
        if new_message_textfield.value != "":
            human_message_text = new_message_textfield.value
            page_width = page.width if page.width > 0 else page.window_width

            if len(chat_listview.controls) == 1:
                chat_listview.controls = []
                await chat_listview.update_async()

            new_message_textfield.value = ""
            await new_message_textfield.update_async()

            await build_message_row(human_message_text, Sender.HUMAN.value, page_width)

            history_messages = await get_lastest_n_records(TURNS)
            bot_message_text = await Bot.get_respond(
                human_message_text, *history_messages, nav_rail.selected_index
            )
            await build_message_row(
                bot_message_text, Sender.BOT.value, page_width, stream=True
            )

            # 存储聊天记录
            records = await page.client_storage.get_async("records")
            records[nav_rail.selected_index] += [[human_message_text, "human"]]
            records[nav_rail.selected_index] += [[bot_message_text, "bot"]]
            await page.client_storage.set_async("records", records)

    async def new_message_textfield_on_focus(e):
        if page.width < page.height:
            if nav_rail.visible:
                await nav_rail_switch_on_click(None)
            page.appbar.visible = False
            await page.appbar.update_async()

    async def new_message_textfield_on_blur(e):
        if page.width < page.height:
            page.appbar.visible = True
            await page.appbar.update_async()

    nav_rail_switch = ft.IconButton(
        icon=ft.icons.KEYBOARD_ARROW_LEFT,
        tooltip="打开/关闭侧栏",
        on_click=nav_rail_switch_on_click,
    )

    clean_chat_records_btn_in_bottom_bar = ft.IconButton(
        icon=ft.icons.DELETE,
        tooltip="清除聊天记录",
        on_click=clean_chat_records,
        visible=False,
    )

    new_message_textfield = ft.TextField(
        label="跟你的病人谈谈吧～",
        hint_text="回车发送消息",
        shift_enter=True,
        min_lines=1,
        max_lines=5,
        expand=True,
        on_submit=send_message_btn_on_click,
        on_focus=new_message_textfield_on_focus,
        on_blur=new_message_textfield_on_blur,
    )

    send_message_btn = ft.FloatingActionButton(
        icon=ft.icons.SEND_ROUNDED,
        tooltip="发送",
        on_click=send_message_btn_on_click,
        visible=False,
    )

    bottom_row = ft.Row(
        [
            nav_rail_switch,
            clean_chat_records_btn_in_bottom_bar,
            new_message_textfield,
            send_message_btn,
        ],
        visible=False,
    )

    ###############################################################
    # ---------------------  控件总体组合与布局 -------------------- #
    ###############################################################
    main_view = ft.Row(
        [
            nav_rail,
            ft.VerticalDivider(width=1),
            ft.Column(
                [
                    ft.Container(
                        content=chat_listview,
                        border=ft.border.all(1, ft.colors.OUTLINE),
                        border_radius=5,
                        padding=10,
                        expand=True,
                    ),
                    bottom_row,
                ],
                expand=9,
                spacing=10,
            ),
        ],
        expand=True,
    )
    await page.add_async(main_view)

    #############################################################
    # -------------------- 不同平台优化 ------------------------- #
    ############################################################
    if page.width < page.height:
        # 窄屏设备
        clean_chat_records_btn_in_appbar.visible = True
        clean_chat_records_btn_in_bottom_bar.visible = False
        await page.update_async()
    elif page.width > page.height:
        clean_chat_records_btn_in_appbar.visible = False
        clean_chat_records_btn_in_bottom_bar.visible = True
        send_message_btn.visible = True
        await page.update_async()

    ##############################################################
    # -------------------- 加载后处理 --------------------------- #
    #############################################################

    if nav_rail.selected_index == None:
        nav_rail.visible = True
        await nav_rail.update_async()
    else:
        # print(nav_rail.selected_index)
        await nav_rail_destinations_on_click(None)

    # 进入上次离开时的视图
    go_to_dict = {
        "nav_rail_leading": nav_rail_leading_on_click,
        "nav_rail_destinations": nav_rail_destinations_on_click,
        "nav_rail_trailing": nav_rail_trailing_on_click,
    }
    await go_to_dict[await page.client_storage.get_async("lastest_view")](None)

    #######################################################
    # ------------- 用于定位错误的统计信息 ----------------- #
    ######################################################

    client_ip, client_platform, client_user_agent = await get_client_info(None)
    print(
        f"""
    Time:{time.ctime()}
    IP:{client_ip}
    Platform:{client_platform}
    User-Agent:{client_user_agent}
    设备上线，上面是加载过程中发生的错误。
    """
    )


ft.app(target=main, view=ft.WEB_BROWSER, port="8550", host="0.0.0.0")
