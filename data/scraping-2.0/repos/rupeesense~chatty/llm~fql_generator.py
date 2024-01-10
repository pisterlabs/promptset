import time

from langchain import HuggingFaceTextGenInference

inference_server_url_local = "http://127.0.0.1:8080"
llm = HuggingFaceTextGenInference(
    inference_server_url=inference_server_url_local,
    temperature=0.1,
)

SYS_PROMPT = '''<s>[INST] Given a user query, you have to identify what information is needed to answer the query.
Include reason on why you need the information to answer the question.

Example:
Input: What is the savings rate for August?
Reason: To answer the question, I need to know the savings rate for August.
Output: savings rate for august

Input: What is my total savings across all bank accounts?
Reason: To answer the question, I need to know the total savings across all bank accounts.
Output: total savings across all bank accounts

Input: How much interest have I earned on my savings over the past year for accountA?
Reason: To answer the question, I need to know the interest earned on savings over the past year for accountA.
Output: interest earned on savings over the past year for accountA

Input: What is my highest and lowest savings month so far this year?
Reason: To answer the question, I need to know the highest and lowest savings month so far this year.
Output: highest and lowest savings month so far this year

Now you have to give reason and output for the following input:
Input: {}
[/INST]
'''

questions = [
    'What percentage of my income am I currently saving?',
    'What is the savings rate for September?',
    'How close am I to reaching my annual savings target?',
    'How much interest will I earn on my current savings over the next year?'
    'How much will I have saved up by the end of the year at my current rate?'
]

for question in questions:
    query = SYS_PROMPT.format(question)
    print(query)
    print(llm.predict(query))
    # sleep for 30 seconds
    time.sleep(30)
