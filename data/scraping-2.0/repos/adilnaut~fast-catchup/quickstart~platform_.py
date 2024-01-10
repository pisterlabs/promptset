import time
import numpy as np
import os
import re
import hashlib
import openai

import tiktoken
from collections import OrderedDict
from datetime import datetime, timedelta
from dateutil import parser
import pytz
import time
import requests
import hashlib
import logging

from retry import retry
from openai.error import RateLimitError
from openai.error import Timeout
from openai.error import APIConnectionError

from PyPDF2 import PdfReader

from quickstart.connection import db_ops, get_platform_id
from quickstart.sqlite_utils import get_upsert_query

def get_abstract_for_slack(slack_message):
    return format_slack_message(slack_message, abstract_func=build_abstract_for_unbounded_text_2), slack_message.ts


def ts_to_formatted_date(ts):
    # dangerous timstamp handling
    return datetime.fromtimestamp(int(ts.split('.')[0])).strftime('%c')

# Opportunity: make it sqllite defined function?
def encapsulate_names_by_ids(text):
    platform_id = get_platform_id('slack')
    if '<@' in text.split('>')[0]:
        left = text.split('>')[0]
        middle = left.split('<@')[1]
        user_data = None
        with db_ops(model_names=['SlackUser']) as (db, SlackUser):
            user_data = SlackUser.query.filter_by(id=middle)  \
                .filter_by(platform_id=platform_id) \
                .first()
        if user_data:
            user_name = user_data.name
            text = text.replace('<@%s>' % middle, user_name)
    return text

def format_slack_message(slack_message, abstract_func=None, date_string=False, channel_misc=False):
    text = slack_message.text
    user_id = slack_message.slack_user_id
    channel_id = slack_message.slack_channel_id
    ts = slack_message.ts

    user_data = None
    platform_id = get_platform_id('slack')
    with db_ops(model_names=['SlackUser']) as (db, SlackUser):
        if platform_id:
            user_data = SlackUser.query.filter_by(id=user_id)  \
                .filter_by(platform_id=platform_id) \
                .first()

    if user_data:
        user_name = user_data.name
        user_email = user_data.profile_email

        channel_data = None
    with db_ops(model_names=['SlackChannel']) as (db, SlackChannel):
        if platform_id:
            channel_data = SlackChannel.query.filter_by(id=channel_id) \
                .filter_by(platform_id=platform_id) \
                .first()

    if channel_data:
        channel_name = channel_data.name
        channel_topic = channel_data.topic
        channel_purpose = channel_data.purpose
        channel_is_channel = channel_data.is_channel
        channel_is_group = channel_data.is_group
        channel_is_im = channel_data.is_im

    # convert ts into datetime formatted string
    date_string = ts_to_formatted_date(ts)

    # encapsulate all mentions to real names by id
    text = encapsulate_names_by_ids(text)
    if len(text) > 50:
        text = abstract_func(text)

    result = 'Slack message:'
    result += ' with text \'%s\' ' % text
    if user_data and user_name:
        result += ' from %s ' % user_name
    if user_data and user_email:
        result += ' with email %s ' % user_email

    # we could also tell if it from channel, group or dm
    if channel_data and channel_is_channel:
        if channel_name:
            result += ' in a channel named %s ' % channel_name
        if channel_topic and channel_misc:
            result += ' with a channel topic %s ' % channel_topic
        if channel_purpose and channel_misc:
            result += ' with a channel purpose %s ' % channel_purpose
    elif channel_data and channel_is_group:
        # could also share num of mebers
        result += ' in a group conversation '
    elif channel_data and channel_is_im:
        result += ' in a direct message conversation '

    if date_string:
        result += ' at %s ' % date_string
    return result


def get_abstract_for_gmail(gmail_message):
    result_text = ""

    id_ = gmail_message.id
    email_ = gmail_message.gmail_user_email
    name_ = None
    snippet_ = None
    final_summary_ = None

    with db_ops(model_names=['GmailUser', 'GmailMessageText']) as \
        (db, GmailUser, GmailMessageText):
        platform_id = get_platform_id('gmail')
        gmail_user = GmailUser.query.filter_by(email=email_) \
            .filter_by(platform_id=platform_id) \
            .one()

        gm_texts = GmailMessageText.query.filter_by(gmail_message_id=id_) \
            .filter_by(is_snippet=False).all()
        summaries = []
        hashes = {}
        for gm_text in gm_texts:
            if gm_text.text_hash not in hashes:
                summaries.append(build_abstract_for_unbounded_text_2(gm_text.text))
            hashes[gm_text.text_hash] = 1
        if len(summaries) == 1:
            final_summary_ = summaries[0]
        else:
            summary = '\n'.join(summaries)
            final_summary_ = build_abstract_for_unbounded_text_2(summary)

        text_hash = hashlib.md5(final_summary_.encode('utf-8')).hexdigest()
        text_kwargs = OrderedDict([('gmail_message_id', id_)
            , ('text_hash', text_hash)
            , ('text', final_summary_)
            , ('is_primary', False)
            , ('is_multipart', False)
            , ('is_summary', True)
            , ('is_snippet', False)])
        text_query = get_upsert_query('gmail_message_text', text_kwargs.keys(), 'gmail_message_id, text_hash')
        db.session.execute(text_query, text_kwargs)

        name_ = gmail_user.name
    subject_ = gmail_message.subject

    result_text += "Email from %s with a subject %s and a summary %s\n" % (name_, subject_, final_summary_)

    return result_text, id_

@retry((Timeout, RateLimitError, APIConnectionError), tries=5, delay=1, backoff=2)
def summarize_with_gpt3(input_text):
    logging.debug('Summarize call with input length %s' % len(input_text))
    time.sleep(0.05)
    ''' Prompt ChatGPT or GPT3 level of importance of one message directly
        TODO: decice where None values should be handled and throw exception
    '''
    openai.api_key = os.getenv("OPEN_AI_KEY")
    # todo might be worth specifying what type of data a bit ( if not independent of metadata )
    prompt = '''Please tell what is the most important in the following text
        and ignore html or other non-text formats: %s''' % input_text

    system_prompt = '''
        You are human work assistant, whose job is to get big chunk of texts
        and to pick the most important points or abstract summaries from text.
        You can either skip details or only get some details from the text,
        depending on what you think is important or what could be urgent.
        Especially if something in a text requires some action.
        If the text contain boilerplate advertisements or news, newsletters -
        you can safely tell that what the text is and give short abstractive summary.
        The most important text is personal, work-related or document related stuff
        - you should give abstractive summary and provide a note that
        there could be more important details and one should skim the whole document.
    '''
    response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
          timeout=50
        )

    text_response = response['choices'][0]['message']['content']
    return text_response

def build_abstract_for_unbounded_text_2(text, truncate=False):
    chunk_length = 3500
    chunk_start = 0
    chunk_end = chunk_length
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = tokenizer.encode(text)
    inputs_batch_lst = []
    while chunk_start <= len(tokens):
        inputs_batch = tokens[chunk_start:chunk_end]
        in_text = tokenizer.decode(inputs_batch)
        inputs_batch_lst.append(in_text)
        chunk_start += chunk_length
        chunk_end += chunk_start + len(inputs_batch)
    summaries = [summarize_with_gpt3(x) for x in inputs_batch_lst]
    summary = '\n'.join(summaries)
    return summary



def test_doc_summary(filepath):
    texts = extract_text_from_pdf(filepath)
    summaries = []
    for text in texts:
        summaries.append(build_abstract_for_unbounded_text_2(text))
    logging.debug(summaries)
    summary = '\n'.join(summaries)
    final_summary = build_abstract_for_unbounded_text(summary)
    logging.debug(final_summary)


def extract_text_from_pdf(filepath):
    # creating a pdf reader object

    reader = PdfReader(filepath)
    # extracting text from page
    text = [page.extract_text() for page in reader.pages]
    return text
