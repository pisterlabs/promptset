import openai #pip install openai
import os

import customtkinter as ctk

def chatGpt():
	prompt = "Please generate some sentences"
	languge = language_dropdown.get()
	prompt += "in " + languge + "."
	
	difficulty = difficulty_value.get()
	prompt += "the level is " + difficulty + "."

	if checkbox1.get():
		prompt += "The sentences could be a question."
	if checkbox2.get():
		prompt += "The sentences could be a negation"

    # Set OpenAI API key
	openai.api_key = os.getenv("OPENAI_API_KEY")
	# Define prompt and parameters
	model = "text-davinci-002"
	temperature = 0.5
	max_tokens = 50

	# Generate text using GPT-3 API
	response = openai.Completion.create(
		engine=model,
		prompt=prompt,
		temperature=temperature,
		max_tokens=max_tokens,
	)
	# Print generated text
	print(response.choices[0].text.strip())
	result.insert("0.0", "\n"+response.choices[0].text.strip())
	res = result.get("0.0", "end-1c")

	# Open a new file in write mode
	with open('answers.txt', 'w') as f:
		# Write text to the file
		f.write(res)

root = ctk.CTk()
root.geometry("500x500")
root.title("Chatbot")
ctk.set_appearance_mode("dark")
#Title
title_label = ctk.CTkLabel(root, text="Sentences in English",font=ctk.CTkFont(size=30,weight="bold"))
title_label.pack(pady=(10,10),padx=10)

#main frame
frame = ctk.CTkFrame(root)
frame.pack(fill="x",padx=10)

#language frame
language_frame = ctk.CTkFrame(frame)
language_frame.pack(padx=10,pady=(20,5),fill="both")
language_label = ctk.CTkLabel(language_frame, text="Language",font=ctk.CTkFont(weight="bold"))
language_label.pack()
language_dropdown = ctk.CTkComboBox(language_frame,values=["Spanish","English","Japanese","Chinese"])
language_dropdown.pack(pady=10)

#language frame
difficulty_frame = ctk.CTkFrame(frame)
difficulty_frame.pack(padx=10,pady=(5),fill="both")
difficulty_label = ctk.CTkLabel(difficulty_frame, text="Level",font=ctk.CTkFont(weight="bold"))
difficulty_label.pack()
difficulty_value=ctk.StringVar(value="Easy")
radiobutton1=ctk.CTkRadioButton(difficulty_frame,text="Beginner",variable=difficulty_value,value="Easy")
radiobutton1.pack(side="left",padx=(20,10),pady=10)
radiobutton2=ctk.CTkRadioButton(difficulty_frame,text="Intermediate",variable=difficulty_value,value="Medium")
radiobutton2.pack(side="left",padx=(20,10),pady=10)
radiobutton3=ctk.CTkRadioButton(difficulty_frame,text="Advance",variable=difficulty_value,value="Hard")
radiobutton3.pack(side="left",padx=(20,10),pady=10)

#features frame
features_frame = ctk.CTkFrame(frame)
features_frame.pack(padx=10,pady=(20,5),fill="both")
features_label = ctk.CTkLabel(features_frame, text="Include",font=ctk.CTkFont(weight="bold"))
language_label.pack()
checkbox1=ctk.CTkCheckBox(features_frame,text="Questions")
checkbox1.pack(side="left",padx=(50),pady=10)
checkbox2=ctk.CTkCheckBox(features_frame,text="Negative sentences")
checkbox2.pack(side="left",padx=(50),pady=10)

button = ctk.CTkButton(frame,text="Generate Sentences",command=chatGpt)
button.pack(padx=10,fill="x",pady=(5,20))


result = ctk.CTkTextbox(root,font=ctk.CTkFont(size=15))
result.pack(pady=10,fill="x",padx=10)

root.mainloop()




