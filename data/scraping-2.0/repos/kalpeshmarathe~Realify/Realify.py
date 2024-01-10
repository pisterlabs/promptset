from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import speech_recognition as sr
import win32com.client
import webbrowser
import openai

openai.api_key = "Add Your api Key Here"


speaker = win32com.client.Dispatch("SAPI.SpVoice")


def ai(prompt):
    # openai.api_key = "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"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    print(response["choices"][0]["text"])
    speaker.Speak(response["choices"][0]["text"])


def takecommand():
    r = sr.Recognizer();
    with sr.Microphone() as source:
        r.pause_threshold = 1
        audio = r.listen(source)
        try:
            query = r.recognize_google(audio, language="en-in")
            print(query)
            return query
        except Exception as e:
            return "Some Error Occured . Sorry From SARA"


def get_second_part(string, delimiter):
    parts = string.split(delimiter)
    if len(parts) > 1:
        return parts[1]
    else:
        return None



 cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video (give video Path)


model = YOLO("../Yolo-Weights/yolov8l.pt")



prev_frame_time = 0
new_frame_time = 0


if __name__ == '__main__':
    print('SARA ACTIVATING')
    s = "Hello Say a Word "
    speaker.Speak(s)
while True:
    print("Listening...")
    text = takecommand()
    speaker.Speak(text)

    if ("open website").lower() in text.lower():
        speaker.Speak(f"Opening {get_second_part(text, delimiter='website')} sir....")
        webbrowser.open(f"https://{get_second_part(text, delimiter='website ')}")

    if ("using ai").lower() in text.lower():
        speaker.speak("Ok i will got your command")
        ai(prompt=text)



    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])


            if(f'{classNames[cls]}'.lower() != 'person'):
                if (("this").lower()  in text.lower()):
                    # speaker.Speak(f"Searching {classNames[cls]}  sir....")
                    newtext = text.replace("this",f'{classNames[cls]}')
                    ai(prompt=newtext)



            print(f'{classNames[cls]} ')
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)




    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

