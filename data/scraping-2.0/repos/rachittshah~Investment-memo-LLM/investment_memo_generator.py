import openai
import smtplib
import ssl
import datetime
from email.message import EmailMessage
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
import os

os.environ["OPENAI_API_KEY"] = "" # Enter your OpenAI API key here

# Function to send the investment memo via email, Yahoo isn't supported at this time
def sendEmail(message):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com" # Enter your email provider's SMTP server here - Yahoo doesn't work
    sender_email = input("Sender Email: ")
    receiver_email = input("Reciever Email: ")
    password ='' # hardcoded password for testing
    # password = loadPassword() # Uncomment this line and comment the line above if you want to load the password from a text file
    msg = EmailMessage()
    msg['Subject'] = 'Investment Memo'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content(message)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.send_message(msg)
        print("Email sent!")

def loadAppPassword():
    file = open("password.txt", "r")
    password = file.read()
    file.close()
    return password

# Function to get investment memo as a text file
def saveMemo(message):
    fileName = input("What would you like to name the file?") + ".txt"
    file = open(fileName, "w")
    file.write(message)
    file.close()
    print("Text file saved! at " + fileName)

def prompt(): # Function to prompt the user for input, you can customize the prompt to your liking
    template = """Create an investment memo on {question} and tell me what the company does, their target audience, a url to the company website, and their competitors, no conclusion, the company's most recent networth, a description of their product or service, and then format it to be an investment memo.
    {chat_history}
    """
    prompt_template = PromptTemplate(input_variables=["chat_history","question"], template=template)
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(
        llm=OpenAI(),
        prompt=prompt_template,
        verbose=True,
        memory=memory,
    )

    result = llm_chain.predict(question=input("Input the company name to create an Investment Memo: ")) # You can change the input to whatever you want
    sendEmail(result)
    saveMemo(result)

    option = input("Would you like to know more about the company? (Y/N)")
    if option == "Y":
        refined = llm_chain.predict(question=input("What would you like to know more about? "))
        print(refined)

print("Running Investment Memo Generator...")
prompt()
