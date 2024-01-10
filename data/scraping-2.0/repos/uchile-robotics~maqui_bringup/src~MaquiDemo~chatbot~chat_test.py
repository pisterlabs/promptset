#!/usr/bin/env python3.9
import rospy
import openai
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class chat_test():
    def __init__(self):
        self.subscriber = rospy.Subscriber('/chatbot/input', String, self._callback)
        self.publisher = rospy.Publisher('/chatbot/output', String , queue_size=10)

    def _callback(self, msg):

        openai.api_key = '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'
        model="gpt-3.5-turbo"
        
        chat = msg.data

        messages = [{"role": "system", "content": "You are a robot called maqui, your model is Pepper from Softbank robotics. You give short answers of no more than 50 words."},
                   {"role": "user", "content": chat}]
        reply = openai.ChatCompletion.create(model=model, messages=messages)
        print(reply["choices"][0]["message"]["content"])
        output = String()
        output.data = reply["choices"][0]["message"]["content"]
        self.publisher.publish(output)

def main():

    rospy.init_node('chat_test')
    chat_test()
    rospy.spin()

if __name__ == '__main__':
    main()