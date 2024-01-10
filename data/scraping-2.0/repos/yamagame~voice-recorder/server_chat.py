import openai
import os
import csv
import datetime
from dotenv import load_dotenv
import difflib
import unicodedata

load_dotenv(dotenv_path=".env.local")

openai.organization = os.environ["OPENAI_ORGANIZATION_ID"]
openai.api_key = os.environ["OPENAI_API_KEY"]


def chatGPT(input_prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_prompt}
    ]

    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100
    )

    content = res['choices'][0]['message']['content']
    return content


def stringCompare(str1, str2):
    return difflib.SequenceMatcher(None, unicodedata.normalize('NFKC', str1), str2).ratio()


class History():
    def __init__(self, row, output=None):
        if output == None:
            self.time = datetime.datetime.strptime(row[0], '%Y/%m/%d %H:%M:%S')
            self.input = row[1]
            self.output = row[2]
        else:
            self.time = datetime.datetime.now()
            self.input = row
            self.output = output

    def __str__(self):
        return '%s input:%s output:%s' % (self.time.strftime('%Y/%m/%d %H:%M:%S'), self.input, self.output)


class ChatHistory():
    history: dict

    def __init__(self):
        self.history = {}

    def load(self, historyfile):
        if not os.path.isfile(historyfile):
            return
        with open(historyfile) as f:
            reader = csv.reader(f)
            for row in reader:
                h = History(row)
                self.history[h.input] = h
                print(h)

    def save(self, historyfile):
        with open(historyfile, mode="w") as f:
            writer = csv.writer(f)
            for k in self.history:
                h = self.history[k]
                writer.writerow([h.time.strftime(
                    '%Y/%m/%d %H:%M:%S'), h.input, h.output])

    def findMatch(self, text):
        for k in self.history:
            if stringCompare(text, k) > 0.8:
                return self.history[k].output
        return ""

    def create(self, input_prompt, mode):
        if mode == "chat":
            # すでに回答済みの対話はキャッシュから返す
            content = self.findMatch(input_prompt)
            if content != "":
                print(content)
                return content
            # organization と api_key が設定されていれば ChatGPT に問合せ
            content = "%sですね" % (input_prompt)
            # if openai.organization != "" and openai.api_key != "":
            #     content = chatGPT("次の問い合せに短く回答してください。"+input_prompt)
            # else:
            #     content = "%sですね" % (input_prompt)
            h = History(input_prompt, content)
            self.history[h.input] = h
            print(content)
            return content
        return input_prompt
