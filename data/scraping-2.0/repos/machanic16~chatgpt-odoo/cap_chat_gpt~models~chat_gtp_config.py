# -*- coding: utf-8 -*-

from odoo import api, fields, models, Command, _
from odoo.exceptions import RedirectWarning, UserError, ValidationError, AccessError
from odoo.tools import float_compare, float_is_zero, date_utils, email_split, email_re, html_escape, is_html_empty
from odoo.tools.misc import formatLang, format_date, get_lang
from odoo.osv import expression

from datetime import date, timedelta
from collections import defaultdict
from contextlib import contextmanager
from itertools import zip_longest
from hashlib import sha256
from json import dumps

import ast
import json
import re
import warnings

import openai

class ChatGPTConfig(models.Model):
    _name = "chatgpt.config"
    _description = 'ChatGPT Configuration'
    _rec_name = 'name'

    name = fields.Char('Name', help='Name of Connection',
                       tracking=True, required=True)

    api_key = fields.Char("Api Key", help='Enter the API Key',
                          tracking=True, required=True)
    
    active = fields.Boolean("Active", default=True)
    
    def send_pront(self,promt,max_tokens=50):
        openai.api_key = self.api_key

        completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": promt}
                        ],
                        max_tokens=max_tokens,
                        temperature=1
                        )
      
        return completion   