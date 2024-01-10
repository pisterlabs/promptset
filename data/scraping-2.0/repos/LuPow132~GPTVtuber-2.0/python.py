from tkinter import *
import customtkinter
import time
import openai
import pytchat
import AUTH_KEY

#Random Variable
Running = False
ChatgptModel = 'gpt-3.5-turbo'
openai.api_key = AUTH_KEY.OPENAI_KEY

#Tkinter thing here
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

#Random Bullshit function
def appendTextFile(filename,input):
    with open(filename,"a") as f:
        f.write(input)
        f.close()

def readTextFile(filename):
    with open(filename,"r") as f:
        read = f.read()
        f.close()
        return read

def clearTextFile(filename):
    with open(filename,"w") as f:
        f.write("")
        f.close()

def overwirteTextFile(filename,text):
    with open(filename,"w") as f:
        f.write(text)
        f.close()

def GPTResponsed(text):
    response = openai.ChatCompletion.create(
    model=ChatgptModel,
    messages=[
        {"role": "system", "content": readTextFile("Preprompt.txt") + readTextFile("Conversation_saver.txt")},
        {"role": "user", "content": text}
    ],
    temperature=.7,
    top_p=1,
    frequency_penalty=0.12,
    presence_penalty=0.2
    )     
    print(f'Reply:{response.choices[0].message.content}')
    print(f'*Succes Generate response token spend: {response.usage.total_tokens}')
    return response.choices[0].message.content

#Here wherer the real shit begin      
def run():
    try:
        global chat
        global ChatLabel
        global Responsed
        stream_ID = StreamIDInput.get()
        chat = pytchat.create(video_id=stream_ID)
        print("*Chat is connected!")

        Responsed = customtkinter.CTkLabel(master=Innerframe,text="Responsed",font=("Roboto",28),wraplength=450)
        Responsed.pack()
        ChatLabel = customtkinter.CTkLabel(master=Innerframe,text="Message",font=("Roboto",20),wraplength=400)
        ChatLabel.pack(pady=10)
        RunButton.configure(state="disabled")
        RunButton.configure(text="Running")
        StreamIDInputField.configure(state="disabled")
        StreamIDInputField.configure(show = "*")

        Innerframe.pack(pady=20,padx=20, fill="both",expand=True)

        ChatConnected()
    except Exception as e:
        print(f"Error detect! Error Info:{e}")
    

def ChatConnected():
    if chat.is_alive():
        for c in chat.get().sync_items():
            message = f'{c.author.name}:{c.message}'
            print(message)
            reply = GPTResponsed(message)
            Responsed.configure(text=reply)
            ChatLabel.configure(text=message)

    root.after(10, ChatConnected)

root = customtkinter.CTk()
root.title("GPT Vtube")
root.geometry("500x300")

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20,padx=60, fill="both",expand=True)


Title = customtkinter.CTkLabel(master=frame,text="GPT Vtube Luncher",font=("Roboto",24)).pack()

StreamIdLabel = customtkinter.CTkLabel(master=frame,text="Youtube Stream ID",font=("Roboto",16)).pack(pady=2)

StreamIDInput = StringVar()
StreamIDInputField = customtkinter.CTkEntry(master=frame,textvariable=StreamIDInput,width=500,height=30)
StreamIDInputField.pack(padx=10)

RunButton = customtkinter.CTkButton(master=frame,text="Run",command=run)
RunButton.pack(pady=5)

Innerframe = customtkinter.CTkFrame(master=frame)

root.mainloop()