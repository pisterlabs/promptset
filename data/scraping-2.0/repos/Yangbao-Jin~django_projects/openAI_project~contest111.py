from openai import OpenAI
import os
import time
import json

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

#client=OpenAI()
#client.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)
file1 = client.files.create(
  file=open("What Does This Program Do_ - ACSL Category Descriptions.pdf", "rb"),
  purpose='assistants'
)
file2 = client.files.create(
  file=open("Recursive Functions - ACSL Category Descriptions.pdf", "rb"),
  purpose='assistants'
)
file3 = client.files.create(
  file=open("Bit-String Flicking - ACSL Category Descriptions.pdf", "rb"),
  purpose='assistants'
)
file4 = client.files.create(
  file=open("Prefix_Infix_Postfix Notation - ACSL Category Descriptions.pdf", "rb"),
  purpose='assistants'
)
assistant = client.beta.assistants.create(
    name="ACSL Tutor",
    instructions='''你是ACSL竞赛题目的答题专家，你叫AC能手。你的职责是回答用户ACSL竞赛中contest 1部分中有关recursive，计算机进制转换和阅读伪指令回答问题等问题。ACSL Guide课堂是AC Genesis公司的一个教育产品。
是帮助参加ACSL竞赛的中学生熟悉ACSL的竞赛知识，解答他们的问题。如果问和ACSL竞赛无关的问题，请告诉他这个问题和ACSL竞赛无关并拒绝回答他的问题。
重要！关于bit flickering解方程的训练：
''',
    tools=[{"type": "code_interpreter"},{"type": "retrieval"}],
    model="gpt-4-1106-preview",
    file_ids=[file1.id,file2.id,file3.id,file4.id]
    
)
user_prompt1='''After this program is executed, what is the value of B that is printed if the input values are 50 and 10?

input H, R
B = 0
if H>48 then
    B = B + (H - 48) * 2 * R
    H = 48
end if
if H>40 then
   B = B + (H - 40) * (3/2) * R
   H = 40
end if
B = B + H * R
output B
'''

user_prompt2='''美国国庆日什么时候?'''

user_prompt3 = '''After the following program is executed, what is the final value of C[4]?

A(0) + 0112: A(1) = 41: A(2) = 52
A(3) = 57: A(4) = 77: A(5) = -100
B(0) = 17: B(1) = 34: B(2） = 81
J = 0: K = 0: N = 0
while A(J) > 0
  while B(K) <= A(J)
    C(N） = B(K）
    N = N + 1
    k = k + 1
  end while
  C(N) = A(J): N = N + 1: J = J + 1
end while
C(N) = B(K)
'''

user_prompt5 = '''find f(12) given:

equation = """
f(x)=f(x-2)-3 when x>=10
f(x)=f(2x-10)+4 when 3<=x<10
f(x)=x*x+5 when x<3
"""
'''
user_prompt6 = '''Problem: Find the value of h(13) given the following definition of h:
equation = """
f(x)=f(x-7)+1 when x>5
f(x)=x when 0<=x<=5
f(x)=f(x+3) when x<0
"""
'''
user_prompt8 = '''你能根据伪码的语法，随机给我生成题目让我练习回答吗？你能对问题的难度分级吗？分为简单，中等，困难三个等级，其中生成的困难题目的代码文本的实际行数必须要大于30行，每个困难等级生成一道题
'''
user_prompt9 = '''Problem: Find the value of f(12,6) given the following definition of f:
equation = """
f(x,y)=f(x-y,y-1)+2 when x>y
f(x,y)=x+y otherwise

"""
'''
user_prompt10 = '''Evaluate the following expression:
()里面的先计算，优先级最高
((RCIRC-14 (LCIRC-23 01101)) | (LSHIFT-1 10011) & (RSHIFT-2 10111))
'''
user_prompt11 = '''List all possible values of x (5 bits long) that solve the following equation.
()里面的先计算，优先级最高。x可以用几位布尔变量代替，列方程，解方程。也可以用python程序来模拟解决，并把结果输出。
(LSHIFT-1 (10110 XOR (RCIRC-3 x) AND 11011)) = 01100
'''
user_prompt12 = '''Evaluate the following expression:
()里面的先计算，优先级最高。计算结果是几位就是几位，不要省略最高位。
(RSHIFT-1 (LCIRC-4 (RCIRC-2 01101))) 
'''
user_prompt13 = '''Evaluate the following expression:
()里面的先计算，优先级最高
(101110 AND NOT 110110 OR (LSHIFT-3 101010))

'''
user_prompt14 = '''Convert the following sequence 
(X = (((A * B) - (C / D)) ↑ E)) from infix to prefix notation.
其中 X和=号也是运算符，参与变换 ()里面的先计算，优先级最高
'''
user_prompt15 = '''Convert the following sequence 
↑ + * 3 4 / 8 2 – 7 5 from prefix to infix notation.
其中 X和=号也是运算符，参与变换 ()里面的先计算，优先级最高
'''

thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=user_prompt15
)


run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
  instructions="if the user ask questions unrelated to the ACSL questions,please refuse to answer.如果他问和ACSL竞赛伪码无关的问题，请告诉他这个问题和ACSL竞赛无关并拒绝回答他的问题"
)
while True:
    run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id)
    if run.status == "completed":
        break
    
#time.sleep(10)   

messages = client.beta.threads.messages.list(thread_id=thread.id)
first_message_value = messages.data[0].content[0].text.value
#first_message_value = messages.data[0].content[0]
print(first_message_value)