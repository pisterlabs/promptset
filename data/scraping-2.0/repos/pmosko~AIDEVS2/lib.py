import requests
import urllib
import functools
import json
import os
import openai
from pprint import pprint
from dotenv import load_dotenv
import argparse

load_dotenv()

TOKEN = os.environ['AIDEV_TOKEN']
URL = "https://zadania.aidevs.pl"

openai.api_key = os.environ['OPENAI_TOKEN']

lib_parser = argparse.ArgumentParser()
lib_parser.add_argument('-o', '--overload', help='overload parameters', action='store_true')
lib_parser.add_argument('-t', '--show-task', help='show task', action='store_true')
lib_parser.add_argument('-r', '--show-result', help='show result', action='store_true')
lib_parser.add_argument('-v', '--verbose', help='verbose', action='store_true')
lib_args = lib_parser.parse_args()
if lib_args.verbose:
    lib_args.show_result = True
    lib_args.show_task = True

def auth_task(taskname: str) -> str:
    """Get token for task"""
    url = get_url('token', taskname)
    load = {
        "apikey": TOKEN,
    }
    return json.loads(requests.post(url=url, json=load).text)['token']

def get_task(token: str=None, task_name: str=None) -> dict:
    """Get Task via token or task name"""
    if not token:
        token = auth_task(task_name)
    url = get_url('task', token)
    return json.loads(requests.get(url=url).text)

def get_url(endpoint, token):
    """Get enpoint url"""
    return functools.reduce(urllib.parse.urljoin, (URL, f'{endpoint}/', token))

def send_task(token: str=None, question: dict=None, headers: dict=None) -> dict:
    """Send Task query"""
    url = get_url('task', token)
    return json.loads(requests.post(url=url, headers=headers, data=question).text)

def send_answer(token: str, answer: dict, headers: dict=None) -> dict:
    """Send Task answer"""
    if lib_args.verbose:
        print("> Anser:")
        pprint(answer)
    url = get_url('answer', token)
    return json.loads(requests.post(url=url, headers=headers, json=answer).text)

def answer(answer=None, *args, **kwargs):
    """Parse response to JSON response format"""
    response = {}
    name = kwargs.pop('key', 'answer')
    if answer is not None:
        response[name] = answer
    for arg in args:
        response.update(*arg)
    response.update(**kwargs)
    return response

class Task:
    def __init__(self, 
                 task_name: str, 
                 send_response: bool = True, 
                 get_task: bool = True, 
                 show_result:bool = False, 
                 show_task: bool = False) -> None:
        self.task_name = task_name
        self.answer = None
        self.question = None
        self.answer_send = False
        if lib_args.overload:
            self.show_result = lib_args.show_result
            self.show_task = lib_args.show_task
        else:
            self.show_result = show_result
            self.show_task = show_task
        self.send_response = send_response
        if get_task:
            self.get_task()

    def get_task(self) -> dict:
        """Get Task content"""
        self.task_token = auth_task(self.task_name)
        self.content = get_task(self.task_token)
        if self.show_task:
            pprint(self.content)
        return self.content
    
    def send_task(self, question: str=None, headers: dict=None):
        """Send question to Task"""
        self.task_token = auth_task(self.task_name)
        if question is None:
            question = self.question
        self.content = send_task(self.task_token, question=question, headers=headers)
        if self.show_task:
            pprint(self.content)
        return self.content
    
    def send_answer(self, answer: str=None, headers: dict=None) -> dict:
        """Send Task answer"""
        if answer is None:
            answer = self.answer
        self.answer_send = True
        return send_answer(self.task_token, answer=answer, headers=headers)
    
    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        if self.send_response and not self.answer_send:
            result = self.send_answer()
            if self.show_result or result['code'] != 0:
                pprint(result)
        return