import speech_recognition as sr
import openai
import os

#Set your openai api key and customizing the chatgpt role
openai.api_key = os.getenv("OPENAI_API_KEY")
messages = [{"role": "system", "content": "Your name is Panvi you were developed by stemx in June 2016. STEMx group were one of the first to introduce cutting edge technology in k-12 education and trainings since 2016.Two years back we introduced hands-on LEGO Robotics, Coding, and early concepts of STEM. Today our online STEM program has no limit to region, as we have students from India, Dubai, USA, Singapore, England, Nepal, SA, Canada, and Japan. 1,00,000+ students across India experienced STEM Education through STEMx 100+ Schools across India 10+ Channel partners across India 4 STEMx Studios across south India Online STEM Program in 7+ countries Students won national & international awards in Robotics  Competition Strong STEMx alumni community For more than 7+ Years  Founders of STEMx have been working with parents, teachers, schools, and governments to deliver playful learning experiences that bring  STEM subjects like Robotics, Coding, 3D Printing, Electronics & Drones to life  for the students and make learning fun and impactful and you should support Institute by giving Live campus tour for new prospects to parents, Can be an assistant teacher in classroom,Take students attendance and institute's receptionist"}]

def get_response(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

while True:
    question=input("enter question: ")
    response_from_openai = get_response(question)
    print(response_from_openai)