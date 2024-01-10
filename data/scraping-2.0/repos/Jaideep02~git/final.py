import speech_recognition as sr
import datetime
import time
import pygame
import threading
import os
import mysql.connector
import tkinter as tk
import webview
import speedtest
import sqlite3
import openai
import pyttsx3
from src.gui.sudoku_gui import GUI



# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Set the voice ID to a female voice
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Set up the OpenAI API client
openai.api_key = "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"

# Set up the model and prompt
model_engine = "text-davinci-003"

# Set up the mysql connection
mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="jai25"
)
# Create a cursor object to execute mysql commands
cursor = mydb.cursor()

# Create a connection to the sqlite database
conn = sqlite3.connect('todo_lists.db')

# Create a cursor object to execute sqlite commands
curz = conn.cursor()

# Set up the wake word detection
wake_word = "Hello"

# Initialize the speech recognizer
r = sr.Recognizer()


# Function to speak out text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Define the function to play a sound
def play_sound():
    pygame.init()
    sound = pygame.mixer.Sound('dark-church-organ-trap-melody_80bpm_B_minor.wav')
    sound.play()
    while pygame.mixer.get_busy():
        pygame.time.Clock().tick(10)



# Define the function to open the app
def open_app(app_name):
    try:
        os.startfile(app_name)
    except FileNotFoundError:
        print(f"Could not find {app_name}.")



# Define the function to check the internet speed
def check_internet_speed():
    st = speedtest.Speedtest()
    print("Your download speed is: ", round(st.download() / 1000000, 3),"Mbps")
    print("Your upload speed is: ", round(st.upload() / 1000000, 3),"Mbps")
    speak("Your download speed is: ", round(st.download() / 1000000, 3),"Mbps")
    speak("Your upload speed is: ", round(st.upload() / 1000000, 3),"Mbps")


# Define the function to set a timer
def set_timer(duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        time.sleep(1)
    print("Time's up!")
    play_sound()


def play_sound():
    # Play the alarm sound
    pygame.mixer.init()
    pygame.mixer.music.load("rnme.wav")
    pygame.mixer.music.play()

def set_reminder(duration, reminder_message):
    # Calculate reminder time by adding duration to the current time
    reminder_time = time.time() + duration * 60
    while True:
        current_time = time.time()
        if current_time >= reminder_time:
            # Print the reminder message and play the alarm sound in a separate thread
            print(reminder_message)
            threading.Thread(target=play_sound).start()
            break

def set_alarm(alarm_time):
    try:
        while True:
            current_time = time.strftime('%H:%M:%S')
            if current_time == alarm_time:
                print("Time's up!")
                pygame.mixer.init()
                pygame.mixer.music.load("alarm_sound.mp3")
                pygame.mixer.music.play()
                break
    except KeyboardInterrupt:
        print("Couldn't set the alarm.")
        speak("Couldn't set the alarm.")

# Define the function to create a sudoku puzzle
def sudoku():
    root = tk.Tk()
    root.geometry("1690x845")
    root.configure(background='#c6c8df')
    root.title("Sudoku Puzzle")

    Game = GUI(root)
    Game.generate_sudoku_board()
    Game.right_side_option_block()

    root.mainloop()


# Define a function to create a new to-do list
def create_todo_list(name):
    # Create a new table for the to-do list
    curz.execute(f"CREATE TABLE {name} (id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT, status TEXT)")
    conn.commit()
    print(f"New to-do list '{name}' created successfully!")
    speak(f"New to-do list '{name}' created successfully!")


# Define a function to add a task to a to-do list
def add_task(todo_list, task):
    # Insert the new task into the specified to-do list
    curz.execute(f"INSERT INTO {todo_list} (task, status) VALUES (?, ?)", (task, 'incomplete'))
    conn.commit()
    print(f"Task '{task}' added to '{todo_list}' successfully!")
    speak(f"Task '{task}' added to '{todo_list}' successfully!")


# Define a function to edit a task in a to-do list
def edit_task(todo_list, task_id, new_task):
    # Update the specified task in the specified to-do list
    curz.execute(f"UPDATE {todo_list} SET task=? WHERE id=?", (new_task, task_id))
    conn.commit()
    print(f"Task {task_id} updated to '{new_task}' in '{todo_list}' successfully!")
    speak(f"Task {task_id} updated to '{new_task}' in '{todo_list}' successfully!")


# Define a function to delete a task from a to-do list
def delete_task(todo_list, task_id):
    # Delete the specified task from the specified to-do list
    curz.execute(f"DELETE FROM {todo_list} WHERE id=?", (task_id,))
    conn.commit()
    print(f"Task {task_id} deleted from '{todo_list}' successfully!")
    speak(f"Task {task_id} deleted from '{todo_list}' successfully!")


# Define a function to delete a to-do list
def delete_todo_list(name):
    # Delete the specified to-do list
    curz.execute(f"DROP TABLE {name}")
    conn.commit()
    print(f"To-do list '{name}' deleted successfully!")
    speak(f"To-do list '{name}' deleted successfully!")


# Define a function to view all the to-do lists
def show_todo_lists():
    # Show all the to-do lists
    curz.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = curz.fetchall()
    print("Tables in the database:")
    speak("Tables in the database:")
    for table in tables:
        print("- " + table[0])
        speak("- " + table[0])


# Define a function to view all the tasks in a to-do list
def show_tasks(list_name):
    curz.execute(f"SELECT * FROM {list_name}")
    tasks = curz.fetchall()
    print(f"Tasks in {list_name}:")
    for task in tasks:
        print(f"- {task[0]}")
        speak(f"- {task[0]}")


# Define the function to create a mysql database
def create_database(database_name):
    try:
        # Use the cursor to execute a mysql command to create the database
        cursor.execute(f"CREATE DATABASE {database_name}")
        # Close the cursor and mysql connection
        cursor.close()
        mydb.close()
    except:
        print("Database already exists.")
        speak("Database already exists.")
        pass


# Define the function to create a mysql table
def create_table(database_name, table_name):
    try:
        cursor.execute(f"USE {database_name}")
        # Use the cursor to execute a mysql command to create the table
        cursor.execute(f"CREATE TABLE {table_name} (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), age INT)")
        print(f"Table '{table_name}' created successfully in database '{database_name}'")
        speak(f"Table '{table_name}' created successfully in database '{database_name}'")
        # Close the cursor and mysql connection
        cursor.close()
        mydb.close()

    except mysql.connector.Error as error:
        print(f"Failed to create table: {error}")
        speak(f"Failed to create table: {error}")

# Define the function to insert a record into a mysql table
# Function to execute commands
def execute_command(command):
    try:
        # Split command
        command_parts = command.split()

        # Extract table name and data
        table_name = command_parts[2]
        data = command.split("set ")[1]
        data = data.replace("'", "")
        data = data.split(",")
        data = [x.strip() for x in data]

        # Create query
        query = f"UPDATE {table_name} SET "
        for item in data:
            item_parts = item.split("=")
            query += f"{item_parts[0]} = '{item_parts[1]}', "
        query = query[:-2] # remove the last comma and space
        query += f" WHERE {command_parts[4]} = '{command_parts[6]}'"

        # Execute query
        cursor.execute(query)
        mydb.commit()

        print("Data updated successfully!")

    except Exception as e:
        print("An error occurred while executing the command:", e)


# Continuously listen for commands
while True:
    # Use the speech recognizer to convert speech to text
    try:
        with sr.Microphone() as source:
            print("Say something!")
            speak("Say something!")
            audio = r.listen(source)
            command = str(r.recognize_google(audio)).lower()

    # If the speech is not recognized, ask for user input
    except sr.UnknownValueError:
        print("Could not understand audio, please input command ~")
        speak("Could not understand audio, please input command ~")
        command = input().lower()

    # If the microphone is not detected, ask for user input
    except sr.RequestError as e:
        print("Could not detect your voice, please try again")
        print(f"Could not request results from Google Speech Recognition service; {e}")
        speak("Could not detect your voice, please try again")
        speak(f"Could not request results from Google Speech Recognition service; {e}")
        command = input().lower()

    except:
        print("Microphone not detected, please input command ~")
        speak("Microphone not detected, please input command ~")
        command = input().lower()

    # Check if the wake word is detected
    if "hello" == command[0:5]:
        # Convert the speech to text or user input
        try:
            print("Please wait... While we process your input...")
            print("You said: ", command)
            speak("Please wait... While we process your input...")
            speak("You said: "+str(command))


            # Check if the command is to open an app
            if "open" in command:
                app_name = command.split("open ")[1]
                open_app(app_name)


            # Check if the command is to set a reminder
            elif "reminder" in command:
                reminder_str = command.split("reminder ")[1]
                time_str = int(command.split("at ")[1])
                set_reminder(time_str, reminder_str)


            # Check if the command is to set a timer
            elif "timer" in command:
                duration_str = command.split("for ")[1]
                duration = int(duration_str)
                set_timer(duration)


            # Check if the command is to search on google
            elif "search" in command:
                search_str = command.split("search ")[1]
                search_url = "https://www.google.com/search?q=" + search_str
                # create a new window
                window = webview.create_window('Search Results', url=search_url, width=1900, height=1000)
                # show the window
                webview.start()


            # Check if the command is to search on youtube
            elif "video" in command:
                search_str = command.split("video ")[1]
                search_url = "https://www.youtube.com/results?search_query=" + search_str
                # create a new window
                window = webview.create_window('Search Results', url=search_url, width=1900, height=1000)
                # show the window
                webview.start()


            # Check if the command is to search on spotify
            elif "song" in command:
                search_str = command.split("song ")[1]
                search_url = "https://open.spotify.com/search/" + search_str
                # create a new window
                window = webview.create_window('Search Results', url=search_url, width=1900, height=1000)
                # show the window
                webview.start()

            # Check if the command is to search on wikipedia
            elif "wiki" in command:
                search_str = command.split("wiki ")[1]
                search_url = "https://en.wikipedia.org/wiki/" + search_str
                # create a new window
                window = webview.create_window('Search Results', url=search_url, width=1920, height=1080)
                # show the window
                webview.start()

            # Check if the command is to play sudoku
            elif "sudoku" in command:
                sudoku()
                print("Opening the game...")
                speak("Opening the game...")


            # Check if the command is to speedtest
            elif "speedtest" in command:
                check_internet_speed()


            # Check if the command is use a to-do list    
            elif "todo" in command:
                if "create" in command:
                    todo_list_name = command.split("named ")[1]
                    create_todo_list(todo_list_name)
                elif "add" in command:
                    todo_list_name = command.split("to ")[1]
                    todo_item = command.split("add ")[1]
                    add_task(todo_list_name, todo_item)
                elif "delete" in command:
                    try:
                        if "from" in command:
                            todo_list_name = command.split("from ")[1]
                            todo_item = int(command.split("delete ")[1])
                            delete_task(todo_list_name, todo_item)
                        else:
                            todo_item = int(command.split("delete ")[1])
                            delete_task("todo", todo_item)
                    except:
                        print("No task/list to delete")
                        speak("No task/list to delete")
                elif "show" in command:
                        if "from" in command:
                            todo_list_name = command.split("from ")[1]
                            show_tasks(todo_list_name)
                        else:
                            show_todo_lists()

                elif "edit" in command:
                    todo_list_name = command.split("from ")[1]
                    todo_item = int(command.split("edit ")[1])
                    new_item = command.split("to ")[1]
                    edit_task(todo_list_name, todo_item, new_item)



            # Check if the command is to create a mysql database or table
            elif "database" in command or "table" in command:
                
                # Check if the next word after "create" is "database" or "table"
                if "create" in command and "table" in command:
                    # The command is to create a table
                    database_name = command.split("database ")[1]
                    table_name = command.split("table ")[1]
                    create_table(database_name, table_name)
                    print(f"Table {table_name} created successfully!")
                    speak(f"Table {table_name} created successfully!")
                elif "create" in command and "database" in command:
                    # The command is to create a database
                    database_name = command.split("database ")[1]
                    create_database(database_name)
                    print(f"Database {database_name} created successfully!")
                    speak(f"Database {database_name} created successfully!")

                elif "show" in command or "all" in command:
                    # The command is to show all databases or tables
                    if "database" in command:
                        # The command is to show all databases
                        cursor.execute("SHOW DATABASES")
                        ctr=0
                        for x in cursor:
                            ctr+=1
                            print(f"Database",ctr,x[0])
                            speak(x[0])
                    elif "tables" in command:
                        # The command is to show all tables
                        database_name = command.split("from ")[1]
                        cursor.execute("USE {}".format(database_name))
                        cursor.execute("SHOW TABLES")
                        tables = cursor.fetchall()

                        # Printing the output
                        print("Tables in {} database:".format(database_name))
                        print("{:<20}{}".format("Table Name", "Rows"))
                        for table in tables:
                            cursor.execute("SELECT COUNT(*) FROM {}".format(table[0]))
                            row_count = cursor.fetchone()[0]
                            print("{:<20}{}".format(table[0], row_count))
                            speak(table[0])

                elif "delete" in command:
                    # The command is to delete a database or table
                    if "database" in command:
                        cursor.execute("DROP DATABASE " + command.split("database ")[1])
                        print("Database deleted successfully!")
                    elif "table" in command:

                        database_name = command.split("database ")[1]
                        table_name = command.split("table ")[1]

                        cursor.execute("DROP TABLE IF EXISTS " + database_name + "." + table_name)

                        print("Table deleted successfully!")
                elif "edit" in command:
                    # The command is to edit a database or table
                    if "database" in command:
                        database_name = command.split("database ")[1]
                        new_database_name = command.split("to ")[1]

                        cursor.execute("ALTER DATABASE " + database_name + " RENAME TO " + new_database_name)

                        print("Database edited successfully!")
                    elif "table" in command:
                        table_name = command.split("table ")[1]
                        new_table_name = command.split("to ")[1]
                        database_name = command.split("database ")[1]
                        cursor.execute(f"USE {database_name}")
                        cursor.execute("ALTER TABLE " + table_name + " RENAME TO " + new_table_name)
                        print("Table edited successfully!")
                elif "insert" in command:
                    # extract database name, table name, and data from command
                    database_name = command.split("in database ")[1].split()[0]
                    table_name = command.split("insert into table ")[1].split()[0]
                    data = command.split("values (")[1].split(")")[0].split(",")

                    # clean up data and convert to tuple
                    data = tuple([x.strip().replace("'", "") for x in data])

                    # use the database
                    cursor.execute(f"USE {database_name}")

                    # execute the SQL query to insert data
                    query = f"INSERT INTO {table_name} VALUES {data}"
                    cursor.execute(query)

                    # commit the changes to the database
                    mydb.commit()

                    # print success message
                    print(cursor.rowcount, "record inserted.")

                    print("Data inserted successfully!")
                    speak("Data inserted successfully!")
                elif "select" in command:
                    #The command is to select data from a table
                    database_name = command.split("database ")[1]
                    table_name = command.split("from ")[1]
                    data = command.split("data ")[1]
                    data = data.replace("'", "")
                    data = data.replace(" ", "")
                    data = data.split(",")
                    data = [x.strip() for x in data]
                    data = tuple(data)

                    cursor.execute(f"USE {database_name}")
                    cursor.execute("SELECT * FROM " + table_name + " WHERE " + data[0] + " = " + data[1])

                    result = cursor.fetchall()
                    for x in result:
                        print(x)
                elif "update" in command:
                    execute_command(command)

            else:
                prompt = command
                # Generate a response
                completion = openai.Completion.create(
                        engine=model_engine,
                        prompt=prompt,
                        max_tokens=1024,
                        n=1,
                        stop=None,
                        temperature=0.5,
                    )
                response = completion.choices[0].text
                print(response)
                speak(response)

        except sr.UnknownValueError:
            print("Speech recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
    else:
        print("Wake word not detected, please try again.")
    time.sleep(1)

'''
Add a sound
Add a GUI
Add a wake word detection
Add a timer
Add a stopwatch
Add a calculator
Add a weather app
Add sudoku
Add a to-do list
Add google, youtube search
Add spotify
Add ChatGPT
Create database(MYSQL) & table edit it too
Create/Edit/Delete a txt or CSV file'''