# Imports
import pyodbc
import random
from datetime import datetime
import openai





# Creating the connection with the database
conn_str = r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=database\db\mainDb.accdb;'
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()


# Funcion to consult the questions to PROMETEU and check if they are in database
def question(question, textAudio):
    try:
        # Execute a consult
        cursor.execute('SELECT perg FROM question WHERE func = '+"'"+question+"';")
        # Recover the consult data
        rows = cursor.fetchall()
        for row in rows:
            roww = str(row[0])
            if roww in textAudio:
                return True
    except pyodbc.Error as e:
        print(e)


# Funcion to consult the answer that PROMETEU needs to gave
def answer(answer):
    try:
        # Execute a consult
        cursor.execute('SELECT resp FROM answer WHERE func = '+"'"+answer+"';")
        # Recover the consult data
        rows = cursor.fetchall()
        preResponse = []
        cont = 0
        for row in rows:
            cont = cont + 1
            roww = str(row[0])
            preResponse.append(roww)
        cont = cont-1
        # Selects a random response and return
        number = random.randint(0, cont)
        response = preResponse[number]
        return response
    except pyodbc.Error as e:
        print(e)


# Funcion to logs with database
def logs(textAudio, response):
    time=datetime.now() 
    hour = int(time.strftime("%H"))
    minutes = int(time.strftime("%M"))
    seconds = time.strftime("%S")
    day = time.strftime("%d")
    week = time.strftime("%A")
    mounth = time.strftime("%B")
    year = time.strftime("%Y")
    try:
        year = str(year)
        mounth = str(mounth)
        week = str(week)
        day = str(day)
        hour = str(hour)
        minutes = str(minutes)
        seconds = str(seconds)
        # Defines the date
        date = str(year+"/"+mounth+"/"+day+":"+week+"/"+hour+":"+minutes+":"+seconds)
        response = str(response)
        textAudio = str(textAudio)
        cursor.execute("INSERT INTO logs(usuario, jarvis, data) VALUES ('"+textAudio+"','"+response+"','"+date+"');")
        # Save the alterations in the logs tabble
        conn.commit()
    # If haves a exeption the code prints what exeption have
    except pyodbc.Error as e:
        print(e)


# Funcion to consult the questions to PROMETEU and check if they are in database
def simpleQuestion(question, textAudio):
    try:
        # Execute a consult
        cursor.execute('SELECT perg FROM simpleQuestion WHERE func = '+"'"+question+"';")
        # Recover the consult data
        rows = cursor.fetchall()
        for row in rows:
            roww = str(row[0])
            if roww in textAudio:
                return True
    except pyodbc.Error as e:
        print(e)


# Funcion to consult the questions to PROMETEU and check if they are in database
def simpleQuestionPerg(question, textAudio):
    try:
        # Execute a consult
        cursor.execute('SELECT perg FROM simpleQuestion WHERE func = '+"'"+question+"';")
        # Recover the consult data
        rows = cursor.fetchall()
        for row in rows:
            roww = str(row[0])
            if roww in textAudio:
                textAudio = textAudio.replace(roww, "")
                return textAudio
    except pyodbc.Error as e:
        print(e)
