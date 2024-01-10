# This Python file uses the following encoding: utf-8

import openai
def check_reslt(qstion,reference,reslt,api):
    print(qstion)
    print(reslt)
    ask = "问题是:["+qstion+"] 我的参考是:["+reference+"]我的回答是:["+reslt+"]请问我的参考和我的回答是否全面"
    print(ask)
    openai.api_key = api
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
#            {"role": "system", "content": qstion}, # 聊天背景
            {"role": "user", "content":ask},
#            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#            {"role": "user", "content": "Where was it played?"}
        ]
    )
#    mssion = completion.choices[0].message[0].content
#    print(completion)
#    return completion
    mssion = completion.choices[0].message.content
    print(mssion)
    return mssion
