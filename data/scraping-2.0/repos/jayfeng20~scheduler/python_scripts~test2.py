from openai import OpenAI
import os  
import calendar
import datetime
import json 
from dotenv import load_dotenv
load_dotenv()
import numpy

import lib.funcs as f
import mysql.connector

# mysql config
host = os.environ.get("DB_ENDPOINT")
user = os.environ.get("DB_USERNAME")
pwd = os.environ.get("DB_PASSWORD")
db = os.environ.get("DB_NAME")


db_config = {
    "host": host,
    "user": user,
    "password": pwd,
    "database": db,
    # 'ssl_ca': 'python_scripts/global-bundle.pem'
}

# openai config
key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(
  api_key=key,
)

int_convert = {}
for i in range(1, 8):
    int_convert['one'] = 1
    int_convert['two'] = 2
    int_convert['three'] = 3
    int_convert['four'] = 4
    int_convert['five'] = 5
    int_convert['six'] = 6
    int_convert['seven'] = 7


def drop_table():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    use_db_query = f"USE {db}"
    cursor.execute(use_db_query)
    print(f"Using database {db}")
    f.drop_table(cursor)
    print(f"Dropped table")
    cursor.close()
    connection.close()
    print("Connection to AWS RDS closed")


def run(task_name='t3', task_type='project', task_time=2, task_due='seven'):
    task_name = task_name
    task_type = task_type
    expected_time = int(task_time)
    due_in = int_convert[f'{task_due}']
    # due_in = task_due
    
    assert expected_time <= 24 and 1 <= expected_time
    assert due_in >= 1 and due_in <= 7 # due in at least 1 day and at most 1 week
    current_datetime = datetime.datetime.now()

    # start the connectionn
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            print("Connected to AWS RDS")
            
        # Create a cursor to execute SQL queries
        cursor = connection.cursor()
        use_db_query = f"USE {db}"
        cursor.execute(use_db_query)
        print(f"Using database {db}")

      # store data
        f.create(current_datetime, cursor)

        alr_booked = []
        rows1, rows2 = f.retrieve(current=current_datetime, cursor=cursor)
        start_times = [row[2] for row in rows1]
        start_times.extend([row[2] for row in rows2])
        end_times = [row[3] for row in rows1]
        end_times.extend([row[3] for row in rows2])
        for i in range(len(start_times)):
            alr_booked.append(f"{start_times[i]} to {end_times[i]}")
        # print(f'alr_booked: {alr_booked}')

      # TODO: Use alr_booked to display initialize the calendar

        
        system_message = """
        You are a smart and concise assistant who is good at time management.
        You do not deliver code. You only give what users ask you for.
        """

        user_message = f"""
        Can you please find 1 timeslot for me to finish a task of mine?
        Only display the found timeslot and nothing else.
        [time right now]: {current_datetime}
        [total time required]: {expected_time} hours,
        [due in]: {due_in} days.
        Already booked slots are: {', '.join(alr_booked)}.

        What humans do not like: 
        1. Working more than 3 hours in a row
        2. not being able to eat lunch

        Requirements:
        1. The total length of the set of timeslots you found has to equal to {expected_time} hours.
        2. Summarize your conclusion and only show me the timeslots represented 
        as 2 things, first marks the start of the timeslot, second marks the end of the timeslot,
        different timeslots are separated by a newline character.
        3. Don't do what humans don't like
        4. You have to STRICTLY schedule timeslots that do not overlap with already booked timeslots.
        5. Ideally, there are breaks in between timeslots for me to rest.
        6. You can only, STRICTLY schedule between 9am and 5pm.
        7. You are allowed to distribute the workload into smaller slots as long as the 
        sum of the lengths of the slots equals the expected duration of the task, but you don't have to.
        8. you ALWAYS generate a set of timeslots.
        9. Double check your output is lists of python datetime objects
        10. Double check your schedules are 9am and 5pm.

        """
    #    7. Summarize your conclusion and only show me the timeslots represent as a python list yourOutput, yourOutput[0] is a python datetime that marks where the timeslot starts. yourOutput[1] is a python datetime that marks where the timeslot ends.

        completion = client.chat.completions.create(
            model="gpt-4-1106-preview",
        #   response_format={ "type": "json_object" },
            messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
            ],
            top_p=0.1,
            # response_format={ "type": "json_object" }
        )

        # ai's response
        response = completion.choices[0].message.content
        response = response.split('\n')
        timeslots = []
        for timeslot in response:
            timeslot = timeslot.split(', ')
            timeslots.append(timeslot)
        print("ai's response1 is" + str(timeslots))
        new_start_times = []
        new_end_times = []
        for ts in timeslots:
            new_start_times.append(ts[0])
            new_end_times.append(ts[1])
        print(new_start_times, new_end_times)
        for i in range(len(new_end_times)):
            new_start_times[i] = datetime.datetime.strptime(new_start_times[i], "%Y-%m-%d %H:%M:%S")
            new_end_times[i] = datetime.datetime.strptime(new_end_times[i], "%Y-%m-%d %H:%M:%S")
        print(new_start_times, new_end_times)
        f.insert(task_name=task_name, start_times=new_start_times, end_times=new_end_times, cursor=cursor)
        


    except mysql.connector.Error as e:
        print(f"Error connecting to AWS RDS: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("Connection to AWS RDS closed")
        print(start_times, end_times, new_start_times, new_end_times)
        return start_times, end_times, new_start_times, new_end_times

# run()
# drop_table()