import unittest
import json
import os
import openai
import random
import re
import collections
import math

lists = [
    [3,4],
    [1,2,3,4,5,6,7,8,9,9],
    [8,8,8],
    [5,6,7],
    [1,6],
    [33333, 44444, 55555],
    ["a"],
    ["a", "a", "a", "A"],
    ["A", "B", "C", "D", "E", "E"],
    ["AB", "CD", "EF"],
    ["F", "E", "A", "G", "X", "D"],
    ["This", "Is", "A", "Test"],
    ["This is another ", "test"],
    ["Test test test"],
    ["i", "S", "E", "n", "g", "A", "RD"],
    ['d', 'w', 'a', 'r', 'v', 'e', 's'],
    ['h0bb', '1t5'],
    [55.5,332.3],
    [7.7,7.7,7.7,7.7,7.7,7.7,7.7,7.7],
    [55.5, 432.4442, 55.4, 55.4, 9894.3333]
]

dicts = [
    {1:1},
    {1:1, 2:2, 3:3, 4:4 ,5:5, 6:6},
    {"1":"1", "2":"2", "3":"3", "4":"4"},
    {"This is a test": "This is test result", "String here":"String there"},
    {1:"1", 2:"2", 3:"3", 4:"4", 5:"5"},
    {55.5:99.3, 443.112:434.332}
]

strings = [
    "String test one",
    "string",
    "t",
    "tttttttttttt",
    "55555555555555s",
    "9 ( ) f",
    "443+33",
    "55 4.55",
    "string string s s string",
    "55 522 04",
    "      ",
    "        f ",
    "s o m e s t r i n g"
]


var_name = "a_val"

testcases = [{
        'question' : f"split string {var_name} into a list",
        'data' : strings,
        'code' : f"{var_name}.split()",
        'inplace' : False
    },{
        'question' : f"reverse the elements in list {var_name} in place",
        'data' : lists,
        'code' : f"{var_name}.reverse()",
        'inplace' : True
    },{
        'question' : f"remove the last element from list {var_name} in place",
        'data' : lists,
        'code' : f"{var_name}.pop()",
        'inplace' : True
    },{
        'question' : f"sort the list {var_name} in descending order",
        'data' : lists,
        'code' : f"{var_name}.sort(reverse=True)",
        'inplace' : True
    },{
        'question' : f"count the number of unique elements in {var_name}",
        'data' : lists + strings,
        'code' : f"len(set({var_name}))",
        'inplace' : False
    },
    {
        'question' : f"check if there are duplicates in list {var_name}",
        'data' : lists,
        'code' : f"len({var_name}) != len(set({var_name}))",
        'inplace' : False
    },
    {
        'question' : f"check if string {var_name} contains number",
        'data' : lists,
        'code' : f"any(i.isdigit() for i in {var_name})",
        'inplace' : False
    },
    {
        'question' : f'create a list containing consecutive numbers between 0 and 1',
        'data' : [[]],
        'code' :  f"[x for x in range(0, 1)]",
        'inplace' : False
    },
    {
        'question' : f'create a list containing consecutive numbers between 3 and 10',
        'data' : [[]],
        'code' :  f"[x for x in range(3, 10)]",
        'inplace' : False
    },{
        'question' : f'create a list containing consecutive numbers between 0 and 100',
        'data' : [[]],
        'code' :  f"[x for x in range(0, 100)]",
        'inplace' : False
    },{
        'question' : f"Get the element of list {var_name} at index 0",
        'data' : lists,
        'code' : f"{var_name}[0]",
        'inplace' : False
    },{
        'question' : f"Get the character of string {var_name} at index 0",
        'data' : strings,
        'code' : f"{var_name}[0]",
        'inplace' : False
    },{
        'question' : f"Get the keys of dictionary {var_name} as a list",
        'data' : dicts,
        'code' : f"list({var_name}.keys())",
        'inplace' : False
    },{
        'question' : f"Get the values of dictionary {var_name} as a list",
        'data' : dicts,
        'code' : f"list({var_name}.values())",
        'inplace' : False
    },
    
    #Add filler word to end.
    #Change variable names
]

def runtest(testcase, gpt_predict):
    data = testcase['data']
    code = testcase['code']
    total = len(data)
    valid = 0
    for d in data:
        out_val = json.dumps(d)
        exec(f"{var_name} = {out_val}")
        try:
            gold_res = eval(code)
        except:
            gold_res = None
        try:
            gpt_res = eval(gpt_predict)
        except:
            gpt_res = None
        if(testcase['inplace']):
            gold_res = eval(var_name)
            gpt_res = eval(var_name)
        if gpt_res == gold_res:
            valid += 1
        print(gold_res, gpt_res)
    return(total, valid)

#Load API keys saved in keys.txt
keyfile = open("../keys.txt", "r")
line = keyfile.readline()
line = line.strip()
openai.api_key = line

prompts = ["""Input: format number of spaces between strings `Python`, `:` and `Very Good` to be `20`
Output: print('%*s : %*s' % (20, 'Python', 20, 'Very Good'))
###
Input: Remove all items from a dictionary `myDict` whose values are `42`
Output: {key: val for key, val in list(myDict.items()) if val != 42}
###
Input: Convert string '03:55' into datetime.time object
Output: datetime.datetime.strptime('03:55', '%H:%M').time()
###
Input: find intersection data between series `s1` and series `s2`
Output: s1.intersection(s2)
###
Input: get the average of a list values for each key in dictionary `d`)
Output: [(i, sum(j) / len(j)) for i, j in list(d.items())]
###
Input: """,
    """
Input: create a list `listofzeros` of `n` zeros
Output: listofzeros = [0] * n
###
Input: Remove all items from a dictionary `myDict` whose values are `42`
Output: {key: val for key, val in list(myDict.items()) if val != 42}
###
Input: Get duplicate elements from list a_val
Output: [value for value in a_val if value in a_val]
###
Input: find intersection data between series `s1` and series `s2`
Output: s1.intersection(s2)
###
Input: get the average of a list values for each key in dictionary `d`)
Output: [(i, sum(j) / len(j)) for i, j in list(d.items())]
###
Input:  get elements from list `a_val`, that have a field `n` value 30
Output:  [x for x in a_val if x.n == 30]
###
Input: """
]

def format_response(response):
    response_dict = response.to_dict()
    text = response_dict["choices"][0]["text"]
    return text


def open_ai_predict(prompt, question):
    #DOCS: https://beta.openai.com/docs/api-reference?lang=python
    #Engines:
    #davinci
    #curie
    #babbage
    #ada
    prompt += question
    prompt += "\n"
    prompt += "Output:"
    print(prompt)
    response = openai.Completion.create(
        engine="davinci",
        temperature = 1,
        top_p = 1,
        prompt=prompt,
        max_tokens=50,
        stop='\n'
        )
    text = format_response(response)
    return text

total_count = 0
valid_count = 0
idempotent_word = ""
for ts in testcases:
    prompt = prompts[1]
    predicted_text = open_ai_predict(prompt, ts['question']+idempotent_word)
    print("The predicted open_ai text is", predicted_text)
    (total, valid) = runtest(ts, predicted_text)
    total_count += total
    valid_count += valid
    print(total, valid)

print(total_count, valid_count, valid_count/total_count)
# def t1():
#     ans = []
#     for data in testcases[0]['data']:
#         x = data
#         res = eval(testcases[0]['code'])
#         ans.append(res)
#     return ans


# class RunComparison(unittest.TestCase):
#     def testcase1(self):
#         truth = t1()
#         ans = []
#         for data in testcases[0]['data']:
#             ans.append(testcases[0]['question'].replace('a_val', data))

#         print(ans, truth)
