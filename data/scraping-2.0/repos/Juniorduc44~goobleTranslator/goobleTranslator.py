# goobleTranslator v0.1.3
import customtkinter
import openai
from dotenv import load_dotenv
import os
load_dotenv()

#Function to delete last result
def myDelete():
    textbox2.delete("0.0", "end")

#list of languages set from optionmenu_1
Values = ['Amharic', 'Arabic', 'Bengali', 'English', 'French', 'Fulani', 'German', 'Gujarati', 'Hausa', 'Hindi', 'Igbo', 'Japanese', 'Javanese', 'Kannada', 'Korean', 'Malay', 'Malayalam', 'Mandarin Chinese', 'Marathi', 'Polish', 'Portuguese', 'Punjabi', 'Russian', 'Shona', 'Somali', 'Spanish', 'Swahili', 'Tamil', 'Telugu', 'Tigrinya', 'Turkish', 'Urdu', 'Vietnamese', 'Wu Chinese', 'Yoruba', 'Zulu']
GoobleValues = ['Amharic', 'Arabic', 'Bengali', 'English', 'French', 'Fulani', 'German', 'Gujarati', 'Hausa', 'Hindi', 'Igbo', 'Japanese', 'Javanese', 'Kannada', 'Korean', 'Malay', 'Malayalam', 'Mandarin Chinese', 'Marathi', 'Polish', 'Portuguese', 'Punjabi', 'Russian', 'Shona', 'Somali', 'Spanish', 'Swahili', 'Tamil', 'Telugu', 'Tigrinya', 'Turkish', 'Urdu', 'Vietnamese', 'Wu Chinese', 'Yoruba', 'Zulu']

#t2 function to translate what user writes in textbox1
def toggle():
    a = True
    G = textbox2.get("0.0", "end")
    if a == True:
        gbTrans = optionmenu_2.get()
        if gbTrans in goobleValues:
            openai.api_key = os.getenv('OPENAI_API_KEY')
            response2 = openai.Completion.create(
                engine = "text-davinci-003",
                prompt = f"Write out the pronunciation of {G} using an {gbTrans} translated Phonetic Alphabet.",
                temperature = 0.0,
                max_tokens = 1000,
                frequency_penalty=0
            )
            #output the goobled language translation to textbox2
            textbox2.insert("0.0", response2.choices[0].text)
    else:
        pass
#t1 main function to translate what user writes in textbox1
def askGPT():
    gpLang = optionmenu_1.get()
    if gpLang in Values:  
        gpText1 = textbox1.get("0.0", "end")
        openai.api_key = os.getenv('OPENAI_API_KEY')
        response = openai.Completion.create(
            engine = "text-davinci-003",
            prompt = f"translate '{gpText1}' into {gpLang} language.",
            temperature = 0.7,
            max_tokens = 1000,
            frequency_penalty=0

        )
        #output the chosen language translation to textbox2
        textbox2.insert("0.0", response.choices[0].text)
    else:
        pass

#create the app window 
app = customtkinter.CTk()
app.geometry("1105x300")
app.title("Gooble Translator")


#t1 create textbox1 for gooble translation 
textbox1 = customtkinter.CTkTextbox(master=app,width=400)
textbox1.grid(row=0, column=1, padx=(20,0), pady=(20,0), sticky="nsew")
#t2 create text box to output gooble translation of textbox1
textbox2 = customtkinter.CTkTextbox(master=app,width=400)
textbox2.grid(row=0, column=3, padx=(20,0), pady=(20,0), sticky="nsew")

#creates a frame effect in background
frame1 = customtkinter.CTkFrame(master = app)
frame1.grid(row=0, column=2,pady=(20,0), padx=(20,0), sticky="nsew") #master or placement is the "app" defined

#DropMenu for language choice
optionmenu_1 = customtkinter.CTkOptionMenu(frame1, values=['Amharic', 'Arabic', 'Bengali',
                                                             'English', 'French', 'Fulani',
                                                              'German', 'Gujarati', 'Hausa',
                                                               'Hindi', 'Igbo', 'Japanese',
                                                                'Javanese', 'Kannada', 'Korean',
                                                                 'Malay', 'Malayalam', 'Mandarin Chinese',
                                                                  'Marathi', 'Polish', 'Portuguese',
                                                                   'Punjabi', 'Russian', 'Shona',
                                                                    'Somali', 'Spanish', 'Swahili',
                                                                     'Tamil', 'Telugu', 'Tigrinya',
                                                                      'Turkish', 'Urdu', 'Vietnamese',
                                                                       'Wu Chinese', 'Yoruba', 'Zulu'],)
#list of languages appended for function
values = ['Amharic', 'Arabic', 'Bengali', 'English', 'French', 'Fulani', 'German', 'Gujarati', 'Hausa', 'Hindi', 'Igbo', 'Japanese', 'Javanese', 'Kannada', 'Korean', 'Malay', 'Malayalam', 'Mandarin Chinese', 'Marathi', 'Polish', 'Portuguese', 'Punjabi', 'Russian', 'Shona', 'Somali', 'Spanish', 'Swahili', 'Tamil', 'Telugu', 'Tigrinya', 'Turkish', 'Urdu', 'Vietnamese', 'Wu Chinese', 'Yoruba', 'Zulu']
optionmenu_1.grid(pady=10, padx=10)
optionmenu_1.set("Language")

#DropMenu for language choice
optionmenu_2 = customtkinter.CTkOptionMenu(frame1, values=['Amharic', 'Arabic', 'Bengali',
                                                             'English', 'French', 'Fulani',
                                                              'German', 'Gujarati', 'Hausa',
                                                               'Hindi', 'Igbo', 'Japanese',
                                                                'Javanese', 'Kannada', 'Korean',
                                                                 'Malay', 'Malayalam', 'Mandarin Chinese',
                                                                  'Marathi', 'Polish', 'Portuguese',
                                                                   'Punjabi', 'Russian', 'Shona',
                                                                    'Somali', 'Spanish', 'Swahili',
                                                                     'Tamil', 'Telugu', 'Tigrinya',
                                                                      'Turkish', 'Urdu', 'Vietnamese',
                                                                       'Wu Chinese', 'Yoruba', 'Zulu'],)
#list of languages appended for function
goobleValues = ['Amharic', 'Arabic', 'Bengali', 'English', 'French', 'Fulani', 'German', 'Gujarati', 'Hausa', 'Hindi', 'Igbo', 'Japanese', 'Javanese', 'Kannada', 'Korean', 'Malay', 'Malayalam', 'Mandarin Chinese', 'Marathi', 'Polish', 'Portuguese', 'Punjabi', 'Russian', 'Shona', 'Somali', 'Spanish', 'Swahili', 'Tamil', 'Telugu', 'Tigrinya', 'Turkish', 'Urdu', 'Vietnamese', 'Wu Chinese', 'Yoruba', 'Zulu']
GoobleValues.append(values)                                                                                                                                       

#used pack to keep button in place instead of using place which lets it move with window
gpt3Button = customtkinter.CTkButton(master=frame1, text="Enter", command=askGPT).grid(padx=10, pady=10)
#button to delete the label of result
delButton = customtkinter.CTkButton(master=frame1, text="Delete", command=myDelete).grid(padx=25, pady=5)
#dropmenu for Gooble Phonics Translation Language
optionmenu_2.grid(pady=10, padx=10)
optionmenu_2.set("Goobled Language")

#s1;v1,t1 This is to create toggle
switch_1 = customtkinter.CTkButton(master=frame1, text="Gooble Phonics", command=toggle)
switch_1.grid(pady=10, padx=10)

#widget to create placement for the results
label = customtkinter.CTkLabel(master=frame1, text="")
label.grid(pady=5, padx=5)


app.mainloop()