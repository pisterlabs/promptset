import os
from datetime import datetime
import pytz
import requests

# 中央氣象局開放資料平臺 https://opendata.cwb.gov.tw
# 資料擷取使用說明 https://opendata.cwb.gov.tw/opendatadoc/CWB_Opendata_API_V1.2.pdf
opendata_access_token  = { 'Authorization': os.environ.get('CWB_AUTHORIZATION_KEY') }

import openai
openai.api_key = os.environ.get('OPENAI_API_KEY')

def chat_bot(json):
    print(json)
    if 'action' in json['queryResult']:
        action = json['queryResult']['action']
    else:
        action = None

    reply = None
    if action == 'input.unknown':
        # Relay the query text to chatGPT
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages= [ { "role":"user", "content": json['queryResult']['queryText'] } ],
            max_tokens=128,
            temperature=0.5,
        )
        reply = response.choices[0].message.content
    elif action == 'query.weather':
        # parameters from Dialogflow
        params = json['queryResult']['parameters']
        weather    = params['getWeather']
        date       = params['date']
        timeperiod = params['time-period']
        location   = params['getLocation']

        # parameters 'date', 'timeperiod' received from Diaglogflow are in ISO 8601 format
        if timeperiod != '':
            startTime = datetime.strptime(timeperiod['startTime'], '%Y-%m-%dT%H:%M:%S%z')
            endTime   = datetime.strptime(timeperiod['endTime']  , '%Y-%m-%dT%H:%M:%S%z')
        else:
            # from specific date to the end of that day if no time period is specified
            if date != '':
                startTime = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S%z')
            else:
                startTime = datetime.now(pytz.timezone('Asia/Taipei'))
            endTime = startTime.replace(hour=23, minute=59, second=59)

        if weather in ['weather', 'temperature']:
            reply = query_weather(location, startTime, endTime, weather)

    return reply

def query_weather(location, startTime, endTime, mode):
    reply = None
    with requests.Session() as s:
        # 鄉鎮天氣預報-臺灣未來一週天氣預報
        url = 'https://opendata.cwb.gov.tw/api/v1/rest/datastore/{0}'.format('F-D0047-091')
        params = { 'format': 'json', 'elementName': 'MinT,MaxT', 'locationName': '臺北市' }
        if location != '':
            params['locationName'] = location
        if mode == 'weather':
            params['elementName'] += ',Wx,PoP12h'

        r = s.get(url, headers = opendata_access_token, params = params).json()
        try:
            forecasts = r['records']['locations'][0]['location'][0]['weatherElement']
            # filter out and reorder the forecast data
            reports = {}
            for element in forecasts:
                for data in element['time']:
                    fromTime = datetime.strptime(data['startTime'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.timezone('Asia/Taipei'))
                    toTime   = datetime.strptime(data['endTime'  ], '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.timezone('Asia/Taipei'))
                    if toTime < startTime or endTime < fromTime:
                        continue
                    key = data['startTime']
                    if key in reports:
                        reports[key][element['elementName']] = data['elementValue'][0]['value']
                    else:
                        reports[key] = { element['elementName']: data['elementValue'][0]['value']}

            if len(reports) > 0:
                reply = '中央氣象局預測%s:' % params['locationName']
                for time, elements in reports.items():
                    reply += '\n*%s以後%s' % (ptime2nl(time), weather_elements2nl(elements))
        except:
            reply = None

    return reply

def ptime2nl(str):
    try:
        t = datetime.strptime(str, '%Y-%m-%d %H:%M:%S')
        now = datetime.now(pytz.timezone('Asia/Taipei')).replace(tzinfo=None)
        days = t.day - now.day

        if days == 0:
            nl_day = '今({})日'.format(t.day)
        elif days == 1:
            nl_day = '明({})日'.format(t.day)
        elif days == 2:
            nl_day = '後天({}日)'.format(t.day)
        else:
            weekday = [ '週一', '週二', '週三', '週四', '週五', '週六', '週日' ]
            month = ''
            if t.month != now.month:
                month = '%d月' % (t.month)
            nl_day = '%s%d日%s' % (month, t.day, weekday[t.weekday()])

        if t.hour < 5:
            nl_hour = '凌晨%d點' % (t.hour)
        elif t.hour < 12:
            nl_hour = '上午%d點' % (t.hour)
        elif t.hour == 12:
            nl_hour = '中午%d點' % (t.hour)
        elif t.hour < 18:
            nl_hour = '下午%d點' % (t.hour - 12)
        else:
            nl_hour = '晚上%d點' % (t.hour - 12)

        return '%s%s' % (nl_day, nl_hour)

    except ValueError:
        return None

def weather_elements2nl(elements):
    desc = ''
    if 'Wx' in elements:
        desc += '，天氣%s' % elements['Wx']
    if 'PoP12h' in elements:
        pop = elements['PoP12h']
        if pop.isdigit() and pop != '0':
            desc += '，降雨機率{}%'.format(pop)
    if 'MaxT' in elements:
        if 'MinT' in elements:
            desc += '，氣溫%s~%s°' % (elements['MinT'], elements['MaxT'])
        else:
            desc += '，最高%s°' % elements['MaxT']

    return desc
