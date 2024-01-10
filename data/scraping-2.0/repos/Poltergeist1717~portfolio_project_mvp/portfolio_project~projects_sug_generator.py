#!/usr/bin/env python3

import os
import openai
import customtkinter as ctk

#Function for generating propmts
def generate():
    prompt = "Please generate 10 suggestions for coding projects. "
    language = language_dropdown.get()
    prompt += "The programming language is " + language + "."
    difficulty = difficulty_value.get()
    prompt += "The difficulty for the project should be " + difficulty + ". "
    experience_level = experience_value.get()
    prompt += "My experience level is " + experience_level + "." 

    if features_checkbox1.get():
        prompt += "The project should include a database."
    if features_checkbox2.get():
        prompt += "The project should include API."

    #print (prompt)

    openai.api_key = "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"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message['content'] #access to content
    #print(answer)
    result.insert("0.0", answer)


#Function for changing THEME
def change_theme():
    if theme_button.get() == 1:
        ctk.set_appearance_mode("dark")
    else:
        ctk.set_appearance_mode("system")

root = ctk.CTk()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

geometry = root.geometry("{} x {}".format(screen_width, screen_height))

root.title("Projects Suggestions Generator")

#ctk.set_appearance_mode("dark")

title_label = ctk.CTkLabel(root, text="Projects Suggestions Generator",
                           font=ctk.CTkFont(size=20, weight = "bold"))
title_label.pack(padx=10, pady=(10, 10))

#Create the main Frame
frame = ctk.CTkFrame(root)
frame.pack(fill="x", padx=100, pady = 10)

#Create the Theme Frame
theme_frame = ctk.CTkFrame(frame)
theme_frame.pack(padx = 100, pady = (10, 5), fill = "both", expand = True)

theme_button = ctk.CTkSwitch(theme_frame, text= "Change Theme", 
                       font=ctk.CTkFont(size= 15, weight= "normal"), command= change_theme,
                       onvalue = 1, offvalue = 0)
theme_button.pack(anchor = "e", padx=10, pady =(5, 5),  expand = True)

#Create the Language Frame
language_frame = ctk.CTkFrame(frame)
language_frame.pack(padx=100, pady=(25, 7), fill="both")
language_label = ctk.CTkLabel(
    language_frame, text="Programming Language", font=ctk.CTkFont(weight="bold"))
language_label.pack(pady = 9)
language_dropdown = ctk.CTkComboBox(
    language_frame, values= ["HTML", "CSS", "Java", "Javascript", "Python", "C", "C++", "PHP", "C#"]
)
language_dropdown.pack(pady = 5)

#Create the Difficulty Frame
difficulty_frame = ctk.CTkFrame(frame)
difficulty_frame.pack(padx=100, pady=5, fill="both")
difficulty_label = ctk.CTkLabel(
    difficulty_frame, text="Project Difficulty", font=ctk.CTkFont(weight="bold")
)
difficulty_label.pack(pady = 8)
difficulty_value = ctk.StringVar(value="Easy")
radiobutton1 = ctk.CTkRadioButton(
    difficulty_frame, text="Easy", variable = difficulty_value, value="Easy", 
)
radiobutton2 = ctk.CTkRadioButton(
    difficulty_frame, text="Medium", variable = difficulty_value, value="Medium", 
)
radiobutton3 = ctk.CTkRadioButton(
    difficulty_frame, text="Hard", variable = difficulty_value, value="Hard", 
)

radiobutton1.pack(side="left", padx=60, pady=10, expand = True)
radiobutton2.pack(side="left", padx=60, pady=10, expand = True)
radiobutton3.pack(side="left", padx=60, pady=10, expand = True)

#Creating Experience Frame
experience_frame = ctk.CTkFrame(frame)
experience_frame.pack(padx=100, pady=5, fill="both")
experience_label = ctk.CTkLabel(
    experience_frame, text="Experience Level", font=ctk.CTkFont(weight="bold")
)
experience_label.pack(pady = 8)
experience_value = ctk.StringVar(value="Easy")
radiobutton1 = ctk.CTkRadioButton(
    experience_frame, text="Beginner", variable = experience_value, value="Beginner", 
)
radiobutton2 = ctk.CTkRadioButton(
    experience_frame, text="Professional", variable = experience_value, value="Professional", 
)
radiobutton3 = ctk.CTkRadioButton(
    experience_frame, text="Expert", variable = experience_value, value="Expert", 
)

radiobutton1.pack(side="left", padx=60, pady=10, expand = True)
radiobutton2.pack(side = "left", padx=60, pady=10, expand = True)
radiobutton3.pack(side="left", padx=60, pady=10, expand = True)

#Create Feature Frame
features_frame = ctk.CTkFrame(frame)
features_frame.pack(padx=100, pady=10, fill="both")
features_label = ctk.CTkLabel(
    features_frame, text="Features", font=ctk.CTkFont(weight="bold")
)
features_label.pack()
features_checkbox1 = ctk.CTkCheckBox(features_frame, text="Database", text_color="white")
features_checkbox2 = ctk.CTkCheckBox(features_frame, text="API", text_color="white")

features_checkbox1.pack(side ="left", padx= 80, pady= 10, expand = True)
features_checkbox2.pack(side ="left", padx= 80, pady= 10, expand = True)

button = ctk.CTkButton(root, text= "Generate Suggestions", 
                       font=ctk.CTkFont(size= 20, weight="bold"), command= generate)
button.pack(padx=100, fill = "x", pady =(5, 10), expand = True)

result = ctk.CTkTextbox(root, font=ctk.CTkFont(size= 15))
result.pack(padx=100, fill = "x", pady =(5, 9), expand = True)

root.mainloop()
