# -*- coding: utf-8 -*-
'''
Create Date: 2023/07/15
Author: @1chooo(Hugo ChunHo Lin)
Version: v0.0.1
Status: Succeed
'''

import unittest
from unittest.mock import MagicMock
import os
import numpy as np
import json
import pandas as pd
import datetime
import tornado.web
import tornado.ioloop
import asyncio
import threading
import pytz
import pymysql
from flask import Flask, Request, abort
from flask import jsonify, render_template
from flask import url_for, send_from_directory
from flask import redirect, Config, Response, request
from werkzeug.utils import secure_filename
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import ImagemapSendMessage
from linebot.models import TextSendMessage
from linebot.models import ImageSendMessage
from linebot.models import LocationSendMessage
from linebot.models import FlexSendMessage
from linebot.models import VideoSendMessage
from linebot.models import StickerSendMessage
from linebot.models import AudioSendMessage
from linebot.models import ImageMessage
from linebot.models import VideoMessage
from linebot.models import AudioMessage
from linebot.models import TextMessage
from linebot.models import TemplateSendMessage
from linebot.models import QuickReply
from linebot.models import MessageTemplateAction
from linebot.models import PostbackAction
from linebot.models import MessageAction
from linebot.models import URIAction
from linebot.models import QuickReplyButton
from linebot.models import LocationAction
from linebot.models import DatetimePickerAction
from linebot.models import RichMenuSwitchAction
from linebot.models.template import ButtonsTemplate
from linebot.models.template import CarouselTemplate
from linebot.models.template import ConfirmTemplate
from linebot.models.template import ImageCarouselTemplate
from linebot.models.events import FollowEvent
from linebot.models.events import MessageEvent
# import openai
# import langchain
# import liffpy

class TestPackages(unittest.TestCase):
    def test_os(self):
        self.assertIsNotNone(os)

    def test_numpy(self):
        self.assertIsNotNone(np)

    def test_json(self):
        self.assertIsNotNone(json)

    def test_pandas(self):
        self.assertIsNotNone(pd)

    def test_datetime(self):
        self.assertIsNotNone(datetime)

    def test_tornado(self):
        self.assertIsNotNone(tornado)

    def test_asyncio(self):
        self.assertIsNotNone(asyncio)

    def test_threading(self):
        self.assertIsNotNone(threading)

    def test_pytz(self):
        self.assertIsNotNone(pytz)

    def test_pymysql(self):
        self.assertIsNotNone(pymysql)

    def test_flask(self):
        self.assertIsNotNone(Flask)
        self.assertIsNotNone(Request)
        self.assertIsNotNone(abort)
        self.assertIsNotNone(jsonify)
        self.assertIsNotNone(render_template)
        self.assertIsNotNone(url_for)
        self.assertIsNotNone(send_from_directory)
        self.assertIsNotNone(redirect)
        self.assertIsNotNone(Config)
        self.assertIsNotNone(Response)
        self.assertIsNotNone(request)

    def test_werkzeug(self):
        self.assertIsNotNone(secure_filename)

    def test_linebot(self):
        self.assertIsNotNone(LineBotApi)
        self.assertIsNotNone(WebhookHandler)
        self.assertIsNotNone(InvalidSignatureError)
        self.assertIsNotNone(LineBotApiError)

    def test_linebot_models(self):
        self.assertIsNotNone(ImagemapSendMessage)
        self.assertIsNotNone(TextSendMessage)
        self.assertIsNotNone(ImageSendMessage)
        self.assertIsNotNone(LocationSendMessage)
        self.assertIsNotNone(FlexSendMessage)
        self.assertIsNotNone(VideoSendMessage)
        self.assertIsNotNone(StickerSendMessage)
        self.assertIsNotNone(AudioSendMessage)
        self.assertIsNotNone(ImageMessage)
        self.assertIsNotNone(VideoMessage)
        self.assertIsNotNone(AudioMessage)
        self.assertIsNotNone(TextMessage)
        self.assertIsNotNone(TemplateSendMessage)
        self.assertIsNotNone(QuickReply)
        self.assertIsNotNone(MessageTemplateAction)
        self.assertIsNotNone(PostbackAction)
        self.assertIsNotNone(MessageAction)
        self.assertIsNotNone(URIAction)
        self.assertIsNotNone(QuickReplyButton)
        self.assertIsNotNone(LocationAction)
        self.assertIsNotNone(DatetimePickerAction)
        self.assertIsNotNone(RichMenuSwitchAction)
        self.assertIsNotNone(ButtonsTemplate)
        self.assertIsNotNone(CarouselTemplate)
        self.assertIsNotNone(ConfirmTemplate)
        self.assertIsNotNone(ImageCarouselTemplate)
        self.assertIsNotNone(FollowEvent)
        self.assertIsNotNone(MessageEvent)

    # def test_linebot_openai(self):
    #     self.assertIsNotNone(openai)

    # def test_linebot_langchain(self):
    #     self.assertIsNotNone(langchain)

    # def test_liffpy(self):
    #     self.assertIsNotNone(liffpy)

if __name__ == '__main__':
    unittest.main()