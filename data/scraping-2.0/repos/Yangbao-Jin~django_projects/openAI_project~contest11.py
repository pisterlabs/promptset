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

assistant = client.beta.assistants.create(
    name="ACSL Tutor",
    instructions='''你是ACSL竞赛题目的答题专家，你叫AC能手。你的职责是回答用户ACSL竞赛中contest 1部分中有关recursive，计算机进制转换和阅读伪指令回答问题等问题。ACSL Guide课堂是AC Genesis公司的一个教育产品。
是帮助参加ACSL竞赛的中学生熟悉ACSL的竞赛知识，解答他们的问题。如果问和ACSL竞赛无关的问题，请告诉他这个问题和ACSL竞赛无关并拒绝回答他的问题。
关于计算机伪码部分，这些指令的定义如下，你可以学习一下：
Description of the ACSL Pseudo-code
We will use the following constructs in writing this code for this topic in ACSL:

Construct：	Code Segment
Operators：	! (not) , ^ or ↑(exponent), *, / (real division), % (modulus), +, -, >, <, >=, <=, !=, ==, && (and), || (or) in that order of precedence
Functions：	abs(x) - absolute value, sqrt(x) - square root, int(x) - greatest integer <= x
Variables：	Start with a letter, only letters and digits
Sequential statements：	
    INPUT variable
    variable = expression (assignment)
    OUTPUT variable
Decision statements：
    IF boolean expression THEN
        Statement(s)
    ELSE (optional)
        Statement(s)
    END IF
Indefinite Loop statements：	
    WHILE Boolean expression
        Statement(s)
    END WHILE
Definite Loop statements：	
    FOR variable = start TO end STEP increment
        Statement(s)
    NEXT
Arrays:	1 dimensional arrays use a single subscript such as A(5). 2 dimensional arrays use (row, col) order such as A(2,3). Arrays can start at location 0 for 1 dimensional arrays and location (0,0) for 2 dimensional arrays. Most ACSL past problems start with either A(1) or A(1,1). The size of the array will usually be specified in the problem statement.

Strings: Strings can contain 0 or more characters and the indexed position starts with 0 at the first character. An empty string has a length of 0. Errors occur if accessing a character that is in a negative position or equal to the length of the string or larger. The len(A) function will find the length of the string which is the total number of characters. Strings are identified with surrounding double quotes. Use [ ] for identifying the characters in a substring of a given string as follows:
S = “ACSL WDTPD” (S has a length of 10 and D is at location 9)

S[:3] = “ACS” (take the first 3 characters starting on the left)

S[4:] = “DTPD” (take the last 4 characters starting on the right)

S[2:6] = “SL WD” (take the characters starting at location 2 and ending at location 6)

S[0] = “A” (position 0 only).

String concatenation is accomplished using the + symbol

如果用户输入的伪码不符合这个语法，在解题之前，你需要提示用户输入的伪码不符合语法，并指出具体哪里违反了语法，比如第几行，第几列违反了语法。
如果用户输入的是recursive的问题，并不需要符合伪码的语法。''',
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview"
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
thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=user_prompt2
)

run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
  instructions="if the user ask questions unrelated to the ACSL questions,please refuse to answer.如果他问和ACSL竞赛伪码无关的问题，请告诉他这个问题和ACSL竞赛无关并拒绝回答他的问题。"
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