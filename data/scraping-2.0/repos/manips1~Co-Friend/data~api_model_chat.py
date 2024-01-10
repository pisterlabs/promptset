import openai
import json

"""
요청 내용
"""
request_str = 'Create a python problem sentence about print and if.'

code_a = """
first_name = input("Enter your first name: ")
last_name = input("Enter your last name: ")

print(last_name + " " + first_name)
"""

code_b = """
fname = input("Input your First Name : ")
lname = input("Input your Last Name : ")
print ("Hello  " + lname + " " + fname)
"""

request_str2 = """
Code A is:
{}

Code B is:
{}

Code B is a answer code.

Grade Code A.

Please answer this Json format:
'pass': True or False,
'score': 0~100,
'reason':text
""".format(code_a, code_b)

"""
API request & response
"""
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {"role": "user", "content": request_str2}
    ]
    )
# json 파싱
json_object = json.loads(response.__str__())
result_text = json_object['choices'][0]['message']['content']

json_object = json.loads(result_text)

#결과 출력
print(json_object['pass'])
print(json_object['score'])
print(json_object['reason'])