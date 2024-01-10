#!/usr/bin/env python3
# coding=utf-8

import rospy
from std_msgs.msg import String
from urllib import response
import openai
   
# Receive question string callback function
def cbQuestion(msg):
    rospy.loginfo("[Rofunc-ros: Chat] --------------------")
    rospy.loginfo(msg.data)

    global api_key
    openai.api_key = api_key
    prompt = msg.data

    global model_engine
    # completion = openai.Completion.create(
    #     engine = model_engine,
    #     prompt = prompt,
    #     max_tokens=1024,
    #     n=1,
    #     stop=None,
    #     temperature=0.5,
    # )


    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "user", "content": prompt}],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.5,
    )

    if 'choices' in completion:
            if len(completion['choices'])>0:
                response= completion['choices'][0]["message"]["content"]
                global response_pub
                answer_msg = String()
                answer_msg.data = response
                response_pub.publish(answer_msg)
            else:
                response = None
    else:
            response = None
    rospy.logwarn(response)

if __name__ == "__main__":
    rospy.init_node("gpt3_node")

    api_key =  rospy.get_param('~openai/api_key')
    model_engine =  rospy.get_param('~openai/model' , "davinci-instruct-beta-v3")

    rospy.logwarn("ChatGPT: current used model %s",model_engine)

    question_sub = rospy.Subscriber("/chatgpt_ask", String, cbQuestion, queue_size=1)

    response_pub = rospy.Publisher("/chatgpt_answer", String, queue_size=1)

    rospy.logwarn("ChatGPT: I'm ready! Please ask your questions ^_^")
    rospy.spin()
