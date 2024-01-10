#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

def listen_and_publish(publisher, client):
    # 设置麦克风的采样率
    sample_rate = 16000
    # 录音时长为5秒
    duration = 5

    try:
        while not rospy.is_shutdown():

            rospy.loginfo("Listening for speech...")
            audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='float32', blocking=True)
            sd.wait()  # 等待录音结束

            # 将录音保存到文件中
            audio_path = './temp_audio.wav'
            sf.write(audio_path, audio, sample_rate)

            # 打开录音文件
            audio_file = open(audio_path, "rb")

            rospy.loginfo("Recognizing speech...")
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
            text = transcript.text
            rospy.loginfo("You said: " + text)
            
            flag = False
            while not flag:
                confirm = input("Is this correct? (y/n): ")
                if confirm in ["y", "Y", "yes", "Yes"]:
                    publisher.publish(text)
                    rospy.loginfo("Published speech to topic.")
                    flag = True
                elif confirm in ["n", "N", "no", "No"]:
                    rospy.logwarn("Not publishing speech to topic. Please try again.")
                    flag = True
                else:
                    rospy.logwarn("Invalid input. Please try again.")
                    flag = False
            
    except Exception as e:
        rospy.logwarn("Whisper could not understand audio or request failed; {0}".format(e))
    finally:
        audio_file.close()
    

if __name__ == '__main__':
    try:
        # 初始化节点
        rospy.init_node('voice_to_text_node', anonymous=True)
        # 创建发布者
        pub = rospy.Publisher('recognized_speech', String, queue_size=10)
        # 创建OpenAI客户端
        client = OpenAI()
        # 设置循环频率
        listen_and_publish(pub, client)
        
    except rospy.ROSInterruptException:
        pass