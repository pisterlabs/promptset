# libraryes
import os
from PyPDF2 import PdfReader
import openai
import streamlit as st
from dotenv import load_dotenv #to resd the key
load_dotenv()
openai.api_key=os.getenv("API_key")






#functions

#read Txt file
def loadfiles(folder):
    text=""
    folderX=os.path.join(os.getcwd(),folder)#join user's folder with variable's folder
    for articles in os.listdir(folderX):#file
        if articles.endswith('.txt'):#catch the file
            with open(os.path.join(folderX,articles),"r")as letter:#it will close the file automaticlly and it will take all texts s leters
                text=text+letter.read()
    return text





    
#read pdf file
def readPdf_File(pdfFile):
    text=''

    reader = PdfReader(pdfFile)# this function to  take and (read) the file . when it take it ,it will split it into a pages
    for page in reader.pages:
        content =page.extract_text()#detect the text
        if content:
            text +=content
    return text






#Get Qustion function
def GetQustion(Qustion):
    prompt=f"""
    You will be given a qustion ,And I want you to answer it.


    qustion: ####{Qustion}####
    """
    



    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content":prompt,

            },
        ],
    )
    return response["choices"][0]["message"]["content"]





#Get summary function
def GetSummary(text):
    prompt=f"""
    You will be given a text , And I want you to summrize it.


    text: #### {text} ####
    """
    



    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content":prompt,

            },
        ],
    )
    return response["choices"][0]["message"]["content"]








# main function
def main():
    #in terminal:
    
    
    
    
    # choice=int(input(" enter 2 for qustion, 1 for sumrizing text ,or 0 for sumrizing pdf :"))
    # if choice ==1:
    #     folder = str(input("Enter folder's name: "))

    #     textFile=loadfiles(folder)
        

    #     print(GetSummary(textFile))
    #     print()
    # elif choice==0:
    #     pdfFile = str(input("Enter pdf file name: "))

    #     pdfText = readPdf_File(pdfFile)
        
    #     print(GetSummary(pdfText))
    # elif choice==2:
    #     qustion=str(input("Enter a qustion:  "))
    #     print(GetQustion(qustion))
    
    #-----------------------------------------------------------
    
    #with streamlit:
    
    
    
    #page title
    
    st.set_page_config(
        page_title="Summrizer",
        page_icon="📚"
        
    )
    
    #Header
    st.header("Sumrizer App")
    st.write("This app uses OpenAI's GPT-3 to summrize a text or an PDF file")
    st.divider()
    
    #options
    option=st.radio("Select one ",("Text","PDF"))
    if option=="Text":
        user_input=st.text_area("Enter a text")
        Submit_buttoum=st.button("Submit")
        if Submit_buttoum and user_input != "":
            response=GetSummary(user_input)
            st.subheader("The Summary")
            st.markdown(response)
        else:
            st.error("please enter text")
            
            
            
        
    elif option == "PDF":
        uploaded_file = st.file_uploader("Choose a file",type=['pdf'])
        if uploaded_file is not None:
            text=readPdf_File(uploaded_file)
            response=GetSummary(text)
            st.subheader("summary")
            st.markdown(response)
        else:
            st.error("Please upload a pdf file")
            
            
        
        
    
    
    
    
    





    
    

    






# execute code
if __name__== "__main__":
    main()
