import os
import re
import json
import openai
import c_utils
from datetime import timedelta
from dateutil import relativedelta
from dateutil.relativedelta import relativedelta
import dateutil.parser as dparser

os.environ['OPENAI_API_KEY'] = 'secret'
openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_fields(prompt):
    res = openai.Completion.create(
          engine='text-davinci-001',
          temperature=0,
          prompt=prompt)
    return res['choices'][0]['text']


def extract_instrument(instrument_test):
    instrument_question = 'What is the financial instrument?'
    instrument_train = '''secret'''
    instrument_prompt = f'{instrument_question}\n{instrument_train}\n{instrument_test}'
    res = c_utils.strip(extract_fields(instrument_prompt))
    return res


def extract_nominal(nominal_test):
    nominal_question = 'What is nominal currency and amount?'
    nominal_train = '''secret'''
    nominal_prompt = f'{nominal_question}\n{nominal_train}\n{nominal_test}'
    res = c_utils.strip(extract_fields(nominal_prompt))
    currency, amt = res.split(',')[:2]
    res_dict = {'currency':currency, 'amt':amt}
    res_link = {'currency':(nominal_test, currency), 'amt':(nominal_test, amt)}
    return res_dict, res_link


def extract_interest(interest_test):
    interest_question = 'What is the interest rate?'
    interest_train = '''secret'''
    interest_prompt = f'{interest_question}\n{interest_train}\n{interest_test}'
    res = extract_fields(interest_prompt)

    if 'round' in interest_test:
        res = f'{res},rounded'
    else:
        res = f'{res},not rounded'
    rate, interval, rounded = c_utils.strip(res).split(',')[:3]
    res_dict = {'rate': rate, 'interval': interval, 'rounded': rounded}
    res_link = {'rate': (interest_test, rate), 'interval': (interest_test, interval), 'rounded': (interest_test, rounded)}

    return res_dict, res_link


def parse_absolute_time(time_str):
    abs_time = dparser.parse(time_str,fuzzy=True).date()
    return abs_time


def parse_relative_time(time_str):
    time_params = {}
    years = re.search('((?P<years>\d+)\s*year)', time_str)
    months = re.search('((?P<months>\d+)\s*month)', time_str)
    weeks = re.search('((?P<weeks>\d+)\s*week)', time_str)
    days = re.search('((?P<days>\d+)\s*day)', time_str)
    if years:
        time_params = c_utils.merge_dicts(time_params, years.groupdict())
    if months:
        time_params = c_utils.merge_dicts(time_params, months.groupdict())
    if weeks:
        time_params = c_utils.merge_dicts(time_params, weeks.groupdict())
    if days:
        time_params = c_utils.merge_dicts(time_params, days.groupdict())

    rel_time = relativedelta(**{k: int(v) for k, v in time_params.items()}) if time_params else None
    return rel_time


def extract_date(end_str, start_str):
    time_params = {'start_date': None,
                   'end_date': None,
                   'time_diff': None}
    time_params['start_date'] = parse_absolute_time(start_str)
    time_params['end_date'] = None
    time_params['time_diff'] = parse_relative_time(end_str)
    if not time_params['time_diff']:
        time_params['end_date'] = parse_absolute_time(end_str)
        time_params['time_diff'] = relativedelta(time_params['end_date'], time_params['start_date'])
    else:
        time_params['end_date'] = time_params['start_date']+time_params['time_diff']
    time_link = {'start_date': (start_str, time_params['start_date']), 'end_date': ((start_str, end_str), time_params['end_date']), 'time_diff': ((start_str, end_str), time_params['time_diff'])}
    return time_params, time_link


def extract_tokenName(tokenName):
    res_dict = {'tokenName': str(tokenName).strip()}
    res_link = {'tokenName': (tokenName, str(tokenName).strip())}
    return res_dict, res_link


def extract_tokenSymbol(tokenSymbol):
    res_dict = {'tokenSymbol':str(tokenSymbol).strip()}
    res_link = {'tokenSymbol':(tokenSymbol,str(tokenSymbol).strip())}
    return res_dict, res_link


def extract_multiple(multiple_test):
    multiple_question = 'What is the multiple?'
    multiple_train = '''secret'''
    instrument_prompt = f'{multiple_question}\n{multiple_train}\n{multiple_test}'
    res = int(re.search(r'\d+', extract_fields(instrument_prompt)).group())
    res_dict = {'multiple':res}
    res_link = {'multiple':(multiple_test, res)}
    return res_dict, res_link
