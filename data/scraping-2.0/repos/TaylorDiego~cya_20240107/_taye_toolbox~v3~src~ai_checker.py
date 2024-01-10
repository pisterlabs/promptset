"""AI Quiz Checker for AI questions"""

import yaml
import json
from openai import OpenAI


OPENAI_API_KEY = yaml.safe_load(open('..\\data\\inputs\\keys.yaml', 'r', encoding='utf-8'))['OPENAI_API_KEY']

def read_quiz(json_data):
    """Get the questions from a JSON file"""
    with open(json_data, 'r', encoding='utf-8') as file:
        return json.load(file)

def make_quiz(_json):
    """Make the quiz from a JSON file"""
    _quiz = {}
    for _num, _items in _json.items():
        _quiz[_num] = {}
        _quiz[_num]['question'] = _items['question']
        _quiz[_num]['options'] = _items['options']
    return _quiz

def make_answer_key(_json):
    """Make the answer key from a JSON file"""
    _answer_key = {}
    for _num, _items in _json.items():
        _answer_key[_num] = {}
        _answer_key[_num]['question'] = _items['question']
        _answer_key[_num]['options'] = _items['options']
        _answer_key[_num]['answer'] = _items['answer']
        _answer_key[_num]['explanation'] = _items['explanation']
    return _answer_key

def save_materials():
    """Save the quiz and answer materials to a file"""
    pass

def study_materials():
    """get contents for quiz from various files in a local directory"""
    pass

def take_quiz():
    """OpenAI quiz taker"""
    pass

def grade_question():
    pass

def grade_options():
    pass

def grade_materials():
    pass

def grade_quiz():
    pass

def make_report():
    pass

def edit_questions():
    pass

def recheck_questions():
    pass

def main():
    """Run the quiz checker"""
    _raw = read_quiz('..\\data\\outputs\\quiz.json')
    _quiz = make_quiz(_raw)
    _answer_key = make_answer_key(_raw)
    # print(f"\nHere's the quiz: \n{_quiz}")
    # print(f"\nHere's the answer key: \n{_answer_key}")
    # return type(_raw)

if __name__ == '__main__':
    main()
    # questions = read_quiz('..\\data\\outputs\\quiz.json')
    # print(questions)