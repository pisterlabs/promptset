#!/usr/bin/env python3
import rospy
import openai
from mr_voice.msg import Voice
from std_msgs.msg import String


def callback_voice(msg):
    global _voice
    _voice = msg
    print(_voice.text)
    

if __name__ == "__main__":
    rospy.init_node("char_ai")
    
    _voice = None
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    _pub_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)

    openai.api_key = "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"

    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        if _voice is None: continue
        if len(_voice.text) <= 10: continue
        
        messages = [
            {"role": "system", "content": "You are a chatbot cat"},
            {"role": "user", "content": _voice.text}
        ]
        
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=_voice.text,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.6,
        )
        result = response.choices[0].text
        
        _voice = None
        #response = openai.ChatCompletion.create(
        #    model="gpt-3.5-turbo",
        #    messages=messages,
        #)
        #result = response.choices[0].message.content

        print(result)
        _pub_speaker.publish(result)
