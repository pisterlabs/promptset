import customtkinter as ctk
from customtkinter import *
import boto3
import openai

class AudioPlayer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Audio Player")
        self.geometry('800x800')
        self.resizable(False, False)
        self.voiceOptions = [
            "Voice1",
            "Voice2",
            "Voice3"
        ]

        self.speedOptions = [
            '1x',
            '2x',
            '3x',
            '4x'
        ]

        self.drop = ctk.CTkComboBox(master=self, values=self.voiceOptions)
        self.drop.grid(row=0, column=0, padx=10, pady=10)

        self.backBtn = ctk.CTkButton(master=self, text="Back")
        self.backBtn.grid(row=0, column=1, padx=10, pady=10)

        self.playBtn = ctk.CTkButton(master=self, text="Play")
        self.playBtn.grid(row=0, column=2, padx=10, pady=10)

        self.nextBtn = ctk.CTkButton(master=self, text="Next")
        self.nextBtn.grid(row=0, column=3, padx=10, pady=10)

        self.drop = ctk.CTkComboBox(master=self, values=self.speedOptions)
        self.drop.grid(row=0, column=4, padx=10, pady=10)

        self.textArea = ctk.CTkTextbox(master=self)
        self.textArea.grid(row=1, column=0, columnspan=5, ipady=300, padx=30 ,pady=30, sticky='nsew')


if __name__ == "__main__":
    player = AudioPlayer()
    player.mainloop()
