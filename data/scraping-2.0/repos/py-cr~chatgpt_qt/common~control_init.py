# -*- coding:utf-8 -*-
# title           :control_init.py
# description     :控件初始化工具类
# author          :Python超人
# date            :2023-6-3
# link            :https://gitcode.net/pythoncr/
# python_version  :3.8
# ==============================================================================
from common.openai_chatbot import OpenAiChatbot
from PyQt5.QtWidgets import QComboBox
from db.db_ops import SessionOp, ConfigOp
from db.entities import Session
from common.str_utils import is_empty


def signal_connected(obj, name):
    """
    判断信号是否连接，用于判断事件是否已经绑定
    :param obj:        对象
    :param name:       信号名，如 clicked()
    """
    index = obj.metaObject().indexOfMethod(name)
    if index > -1:
        method = obj.metaObject().method(index)
        if method:
            return obj.isSignalConnected(method)
    return False


def init_ai_model_combo(cmb_ai_model: QComboBox, session_id):
    """

    :param cmb_ai_model:
    :param session_id:
    :return:
    """

    cmb_ai_model.clear()
    model_list = OpenAiChatbot().get_model_list()

    for item in model_list:
        cmb_ai_model.addItem(item['id'])

    session: Session = SessionOp.select(session_id, Session)
    model_id = session.model_id
    if is_empty(model_id):
        model_id = "gpt-3.5-turbo"
    cmb_ai_model.setCurrentText(model_id)


def init_ai_role_combo(cmb_ai_role: QComboBox):
    """

    :param cmb_ai_role:
    :return:
    """
    cmb_ai_role.clear()
    roles = ConfigOp.get_ai_roles()
    cmb_ai_role.addItem("", "")
    for role in roles:
        cmb_ai_role.addItem(role.cfg_key, role.cfg_value)


def init_categories_combo(cmb_categories: QComboBox, session_id, category_changed):
    """
    初始化聊天话题分类下拉框控件
    :return:
    """
    if signal_connected(cmb_categories, 'currentIndexChanged()'):
        cmb_categories.currentIndexChanged.disconnect()
    cmb_categories.clear()
    cmb_categories.addItem("", 0)
    for category in ConfigOp.get_categories():
        cmb_categories.addItem(category.cfg_key, category._id)

    session: Session = SessionOp.select(session_id, Session)
    if session is not None:
        index = cmb_categories.findData(session.category_id)  # 查找值为 20 的选项的索引
        cmb_categories.setCurrentIndex(index)  # 将找到的索引设置为当前选中项

    cmb_categories.currentIndexChanged.connect(category_changed)
