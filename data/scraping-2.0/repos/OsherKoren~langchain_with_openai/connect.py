# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""This module is for connecting to the OpenAI API."""

import openai
import os
import serpapi

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.getenv("OPENAI_API_KEY")
serpapi.api_key = os.getenv("SERPAPI_API_KEY")
