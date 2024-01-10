# import speech_recognition as sr
# import openai
#
#
# def get_audio():
#     ear_robot = sr.Recognizer()
#
#     with sr.Microphone() as source:
#         print("Trợ Lý Ảo:  Đang nghe ! -- __ -- !")
#
#         # ear_robot.pause_threshold = 4
#         # audio = ear_robot.record(source , duration= 4)
#         # ear_robot.language = "vi-VN"
#         audio = ear_robot.listen(source)
#         # audio = ear_robot.listen(source, phrase_time_limit=5)
#
#         try:
#             print(("Trợ Lý Ảo :  ...  "))
#             # text = ear_robot.recognize(audio)
#             text = ear_robot.recognize_google(audio, language="vi-VN")
#             # print("Tôi:  ", text)
#             return text
#         except Exception as ex:
#             print("Trợ Lý Ảo:  Lỗi Rồi ! ... !")
#             return ""
#
#
# openai.api_key = "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"
#
# start_sequence = "\nAI:"
# restart_sequence = "\nHuman: "
#
# cau_hoi = input("Speech : ")
#
# response = openai.Completion.create(
#     model="text-davinci-003",
#     prompt=f"The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: {cau_hoi}\nAI: ",
#     temperature=0.9,
#     max_tokens=300,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0.6,
#     stop=[" Human:", " AI:"]
# )
#
# text = response['choices'][0]['text']
# print(text)
# print(f"Response length is : {len(text)}")

print("hello")