import PySimpleGUI as sg
import mysql.connector
import pandas as pd
import os
import openai

openai.api_key = "your_openai_api_key"

#gui window
sg.theme('SandyBeach')
layout = [
    [sg.Text("Please Enter SQL")],
    [sg.Multiline(size=(50,5), key='textbook')],
    [sg.Submit(), sg.Cancel()]
]
window = sg.Window("Report Generator", layout)
event, values = window.read()
query = values['textbook']

host1 = os.environ.get("host")
user1 = os.environ.get("user")
password1 = os.environ.get("password")


#SQL connection/query
cnx = mysql.connector.connect(
    host = host1,
    user = user1,
    password = password1
)

#create and open report
df = pd.read_sql(query, cnx)
df.to_excel('report.xlsx', index=False)
os.startfile('report.xlsx')