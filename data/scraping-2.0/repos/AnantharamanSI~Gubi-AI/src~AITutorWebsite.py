import streamlit
import cv2
import PicToAnswer
import openai
import os
import TextReader

class AITutorWebsite:
    def __init__ (self):
        streamlit.set_page_config(page_title="Gubi")
        streamlit.title("Gubi AI") 
        self.frameHolder = streamlit.empty()
        self.questionScan = True

        streamlit.write('Please note that sometimes ChatGPT may time out. If 20 seconds passes without an update, please click "New Question"')

        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        resetButton = streamlit.button("New Question", 
            on_click=self.resetQuestion,
            disabled=False,
        )

        self.AIClient = openai.OpenAI(api_key=str(os.environ.get("OPENAI_API_KEY")))
        self.mainLoop()


    def resetQuestion(self):
        self.questionScan = True
        self.camera.open(0)


    def mainLoop(self):
        FPS = 30
        INTERVAL = 10
        frameCount = 1
        frameHolder = streamlit.empty()
        question = ""
        wolframAnswer = ""
        studentAnswer = ""
        while(True):
            success, frame = self.camera.read()
            if (success):
                self.displayFrame(frameHolder, frame)
                if (frameCount % (FPS * INTERVAL) == 0 and self.questionScan):
                    question = self.saveAndReadPhoto("question.png", frame)

                    if (question == PicToAnswer.BAD_INPUT_MESSAGE): 
                        frameCount += 1
                        continue

                    self.questionScan = False
                    wolframQuestion = PicToAnswer.wolframLatexParser(question)
                    wolframAnswer = PicToAnswer.getWAanswer(wolframQuestion)

                elif (frameCount % (FPS * INTERVAL) == 0 and not self.questionScan):
                    print("log")
                    studentAnswer = self.saveAndReadPhoto("StudentAnswer.png", frame)
                    if (studentAnswer == PicToAnswer.BAD_INPUT_MESSAGE): 
                        frameCount += 1
                        continue
                    aiComparison = PicToAnswer.getAIAnswer(question, studentAnswer, wolframAnswer, self.AIClient)
                    TextReader.talk(aiComparison)
                    streamlit.write(aiComparison)


            frameCount += 1


    def displayFrame(self, frameHolder, frame):
        frameHolder.image(frame, channels="BGR")


    def getWA(self, imagePath):
        PicToAnswer.getWAanswer(imagePath)

    def saveAndReadPhoto(self, name, frame):
        cv2.imwrite(name, frame)
        rawInput = PicToAnswer.imageToAnswer(name)
        return rawInput