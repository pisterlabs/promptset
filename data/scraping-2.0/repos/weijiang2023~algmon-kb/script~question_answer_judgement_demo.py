'''
POC Demo
使用算法妈妈智能API去批改问题的答案
黄金三元组框架，三元组为（问题，答案，批改）(question, answer, judgement) 
'''
import openai
import pandas as pd

# teacher judgement is used as the ground truth
# 老师批改学生作业模式
# gpt-5 is used
teacher_mode = True
if teacher_mode == True:
    # Read the csv file
    file_path = '../kb/structured/domain.教培/domain.math.grade.5.上/练习2.qa.csv'
    df = pd.read_csv(file_path)
    #print(df.columns)
    #print(df.head())
    qapairs = []
    for index, row in df.iterrows():
        #print(index)
        q = row['question']
        a = row['answer']
        #print(q)
        #print(a)
        qapairs.append("请您作为专家老师帮忙看看以下解答是否正确，并且为这个学生的回答打一个分数，分数范围为0-10，6分为合格。" + "数学问题是:" + q + "。" + "学生的回答是:" + a)
    #print(qapairs)
    print("total # of questions:", len(qapairs))

    #option 1
    #openai.api_key = os.getenv("OPENAI_API_KEY") # 需先在终端设置环境变量 OPENAI_API_KEY
    #openai.api_base = os.getenv("OPENAI_API_BASE") # 需先在终端设置环境变量 OPENAI_API_BASE

    #option 2
    openai.api_key = ''
    openai.api_base = ''

    for idx,qapair in enumerate(qapairs):
        req = qapair
        completion = openai.ChatCompletion.create(
          model="gpt-4",
          messages=[
              {"role": "system", "content": "您是一个数学和英语培优专家 能专业正确地回答各种知识点，问题及解法"},
              {"role": "user", "content": req}
          ]
        )

        #print(completion.choices[0].message.encode('utf-8').decode('unicode_escape'))
        res = completion.choices[0].message
        print("question",idx)
        print("请求（问题与学生）：", req)
        print("回答（老师的金标）：", res["content"])
else:
    # TODO:
    pass