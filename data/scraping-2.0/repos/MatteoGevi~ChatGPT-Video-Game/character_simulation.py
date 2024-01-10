import os
from flask import Flask
import json

from requests import HTTPError
from typing import List
from uuid import uuid4 as uuid

from chatgpt.authentication import OpenAIAuthentication
from chatgpt.sessions import HTTPSession, HTTPTLSSession
from chatgpt.utils import get_utc_now_datetime

gpt_key = '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'

