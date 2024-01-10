from calendar_main import add_event
from datetime import date
import json
import os
import openai
from IPython.display import display, Markdown
import yfinance as yf

def parse_input(user_input, openaiKey):
    """
    This method parse the user input into a json file that can be used to add the calendar event.
    """

    # Set openai.api_key to the OPENAI environment variable
    openai.api_key = openaiKey

    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": '忘记你所有之前的指示。你是一个帮助用户添加谷歌日历事件的小助手。你需要从用户的输入中提取出三个关键信息并生成一个json文件。json文件中包含三个key，分别是summary，startTime，endTime，和timeZone。你需要在Summary中总结出一个简短的日历事件的名字。startTime和endTime分别代表了该事件发生的起始时间和终止时间，其格式为YYYY-MM-DDTHH:MM:SS，例如，2023年11月22日上午9点就是2023-11-22T09:00:00。timeZone代表了前面时间所处的时区，一般你需要在两个选项中选择，分别是Europe/London和Asia/Hong_Kong，如果用户不说的话，默认时区是Europe/London。并且你需要推测出用户所描述的时间，今天的时间是' + str(date.today()) + '。记住使用```json和```来分割json文件。'},
                            {"role": "user", "content": str(user_input) + '。记得采用```json + json代码 + ```的形式来进行输出'}
                ])
    # print(response)
    # print()
    message = response['choices'][0]['message']['content']
    print(message)
    if '```json' in message:
        json_text = message.split('```json')[1].split('```')[0]
    else:
        json_text = message
    print(json_text)
    json_data = json.loads(json_text)

    with open('event_detail.json', 'r') as file:
        event_data = file.read()
    event = json.loads(event_data)

    event['summary'] = json_data['summary']
    event['start']['dateTime'] = json_data['startTime']
    event['start']['timeZone'] = json_data['timeZone']
    event['end']['dateTime'] = json_data['endTime']
    event['end']['timeZone'] = json_data['timeZone']

    with open('event_detail.json', 'w') as file:
        json.dump(event, file, indent=4)
    print('done saving the json file!')

    # print('adding event...')
    # add_event()
    # print('finished!')
