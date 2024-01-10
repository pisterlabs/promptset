"""
filepath: 
功能: 

"""
import json
import xml.etree.ElementTree as ET
import openai
from pprint import pprint

filepath = r'./res/AmazeFileManager.xml'

fewshot = """
Q: Here is an app named "AmazeFileManager", and all its activities are: Preferences, TextEditor, Main, DatabaseViewer. Please give the order of testing for these activities.
A: The order should be Main, Preferences, TextEditor, DatabaseViewer.

"""


def getOutput(question: str):
    """
    
    """
    openai.api_key = "xxx"

    start_sequence = "\nA:"
    restart_sequence = "\n\nQ: "

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=question + "\nA:",
        temperature=0.5,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
    )

    return response.choices[0].text


def main(appname):
    tree = ET.parse(filepath)

    root = tree.getroot()

    app = root.find('application')

    activities = []

    for e in app.iter('activity'):
        if e.find('intent-filter') is not None:
            activities.append(e.attrib['{http://schemas.android.com/apk/res/android}name'].split('.')[-1].replace('Activity', ''))

    print(activities)

    prompt = fewshot
    prompt += 'Q: Here is an app named "{}", and all its activities are: '.format(appname)

    for e in activities:
        prompt += e + ", "

    prompt = prompt[:-2] + '. Please give the order of testing for these activities.'

    print('prompt: {}'.format(prompt))
    res = getOutput(prompt)
    print('res: {}'.format(res))
    real_res = res[20: -1]
    print('real res: {}'.format(real_res))
    order_list = real_res.split(',')
    for i in range(len(order_list)):
        order_list[i] = '{}_{}'.format(i + 1, order_list[i].strip())
    print(order_list)

    jsondata = {
        "Global information attribute": [{
            "App name" : appname,
            "Activities": activities,
            "Priority": order_list
        }]
    }

    print('json data:')
    pprint(jsondata)

    jsonstr = json.dumps(jsondata)
    f = open('./json/{}.json'.format(appname), 'w')
    f.write(jsonstr)
    f.close()


if __name__ == '__main__':
    main('AmazeFileManager')
