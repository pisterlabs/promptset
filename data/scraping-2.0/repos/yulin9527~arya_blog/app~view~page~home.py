from nicegui import ui
from app.view.element.article import editor_car, top_article
from app.view.element.message import openai_msg


async def main_page():
    """
    主页布局
    """
    with ui.header(elevated=True).classes('items-center justify-between'):
        ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat color=white')
        ui.button(on_click=lambda: right_drawer.toggle(), icon='rocket_launch').props('flat color=white')
    with ui.left_drawer(top_corner=False, bottom_corner=False) as left_drawer:
        personal_info()
        openai_msg()

    with ui.right_drawer(fixed=True, top_corner=False, bottom_corner=False).props(
            'bordered').props('width=400') as right_drawer:
        ui.button(on_click=lambda: right_drawer.toggle())
        openai_msg()

    top_article()
    top_article()
    top_article()

    # editor_car()
    #
    # def set_background(color: str) -> None:
    #     ui.query('body').style(f'background-color: {color}')  # 这句是关键，设置body的style
    #
    # ui.image('/static/head_img/arya.jpg').classes('w-16')
    # ui.button('Blue', on_click=lambda: set_background('#ddeeff'))  # 点击后，页面背景颜色变为蓝色
    # ui.button('Orange', on_click=lambda: set_background('#ffeedd'))  # 点击后，页面背景颜色变为橙色
    # # ui.label('CONTENT')
    # [ui.label(f'Line {i}') for i in range(100)]
    #

    #     personal_info()
    #     left_path_home()
    #     essay_class(essay_list)
    #     ui.separator()
    #     ui.separator()
    #     with ui.card():
    #         with ui.row():
    #             ui.button('主页')
    #             ui.button('评论')
    #             ui.button('github')
    #             ui.button('登录')

    # with ui.footer():  # .style('background-color: #3874c8')
    #     ui.label('FOOTER')


def personal_info():
    # 头像小组件
    with ui.card().classes('w-full'):
        with ui.row():
            # ui.avatar('img:/static/head_img/arya.jpg', square=True, rounded=False, size='xl').classes('no-margin')
            ui.image('/static/head_img/arya.jpg').classes('w-16').style('border-radius:10px')
            # ui.image('/static/head_img/arya.jpg').classes('w-16').style('border-radius:10px')
            with ui.column():
                ui.label('羽林').style(
                    'font-weight:bold; font-size:18px;width:100%; text-align: center;')
                ui.label('What are you prepared to do?').style('font-size:12px;font-style:italic')


essay_list = ['分类1', '分类 2', '分类 3'] * 30


def essay_class(essay_list):
    with ui.card().tight().classes('w-full'):
        with ui.expansion('分类!', icon='work'):
            for essay in essay_list:
                ui.label(essay)
        ui.separator()
        with ui.expansion('页面', icon='work'):
            pass


def left_path_home():
    with ui.card().tight().classes('w-full'):
        ui.label('首页')
        ui.label('相册')
        ui.label('日记')
        ui.label('关于')

