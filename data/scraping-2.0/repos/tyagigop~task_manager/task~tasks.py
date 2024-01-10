
from twilio.rest import Client
import sqlite3

from django.utils import timezone
from datetime import timedelta
import pytz
import openai


openai.api_key = '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'


# client = OpenAI(api_key='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')


# Set your OpenAI API key


from celery import shared_task


account_sid = 'xxxxx'
auth_token = 'xxxxx'
client = Client(account_sid, auth_token)



def handle_user_response(request, user_number, user_message):
    conn = sqlite3.connect('usertasks.db',check_same_thread=False)
    print(user_message)
    original_message = user_message
    user_message = user_message.lower()
    if user_message == 'add task' or user_message == 'add' or user_message == 'add tasks':
        link = "https://taskblaze.tech/add_tasks/"+user_number[12:]
        client.messages.create(
            from_='whatsapp:+917618207974',
            body=link,
            to=user_number
        )
    elif user_message == 'view tasks' or user_message == 'view task' or user_message == 'view':
        link = "https://taskblaze.tech/view_tasks/"+user_number[12:]
        c = conn.cursor()
        c.execute("SELECT * FROM Users WHERE mobile_number = ?", (user_number[12:],))
        user = c.fetchone()
        user_id = user[0]
        current_date = timezone.now().strftime('%d-%m-%Y')
        c.execute("SELECT * FROM Tasks WHERE user_id = ? AND date = ?", (user_id, current_date))
        
        tasks = c.fetchall()
        c.close()
        if not tasks:
            client.messages.create(
                from_='whatsapp:+917618207974',
                body="You don't have any tasks for today.",
                to=user_number
            )
            return
        message = "🌟 *Your Tasks for Today* 🌟\n\n"
        incomplete_tasks = 0
        completed_tasks = 0
        for task in tasks:
            task_name = task[2]
            task_time = task[4]
            task_status = task[8]
            if task_status == 'Incomplete':
                incomplete_tasks += 1
            else:
                completed_tasks += 1
            status_icon = "✅" if task_status == 'Complete' else "❌"
            message += f"*{task_name}*\nTime: {task_time}\nStatus: {status_icon} {task_status}\n\n"

            
        
        completion_percentage = (completed_tasks/(completed_tasks+incomplete_tasks))*100
        message += f"🚀 You have completed {completion_percentage:.2f}% of your tasks for today.\n\n"
        message += f"👉 You View Tasks Here :\n\n{link}"
        client.messages.create(
            from_='whatsapp:+917618207974',
            body=message,
            to=user_number
        )
    elif user_message == 'help':
        client.messages.create(
            from_='whatsapp:+917618207974',
            body="Type 'add task' to add a task. Type 'view tasks' to view your tasks.",
            to=user_number
        )
    elif user_message == 'hi' or user_message == 'hello' or user_message == 'hey' or user_message == 'hii' or user_message == 'hiii' or user_message == 'hiiii' or user_message == 'hiiiii' or user_message == 'hiiiiii' or user_message == 'hiiiiiii' or user_message == 'hiiiiiiii' or user_message == 'hiiiiiiiii' or user_message == 'hiiiiiiiiii' or user_message == 'hiiiiiiiiiii' or user_message == 'hiiiiiiiiiiii' or user_message == 'hiiiiiiiiiiiii' or user_message == 'hiiiiiiiiiiiiii' or user_message == 'hiiiiiiiiiiiiiii' or user_message == 'hiiiiiiiiiiiiiiii' or user_message == 'hiiiiiiiiiiiiiiiii' or user_message == 'hiiiiiiiiiiiiiiiiii' or user_message == 'hiiiiiiiiiiiiiiiiiii':
        client.messages.create(
            from_='whatsapp:+917618207974',
            body="Hey! Type 'help' to know more.",
            to=user_number
        )
    elif user_message == 'yes' or user_message == 'done' or user_message == 'completed' or user_message == 'finished':
        c = conn.cursor()
        c.execute("SELECT * FROM Users WHERE mobile_number = ?", (user_number[12:],))
        user = c.fetchone()
        user_id = user[0]
        c.execute("SELECT * FROM \"Reminder Responses\" WHERE user_id = ?", (user_id,))
        reminders = c.fetchall()
        
        # Assuming index 4 represents the date and time in the format 'YYYY-MM-DD HH:MM:SS'
        # task_id = None
        sorted_reminders = sorted(reminders, key=lambda x: x[4], reverse=True)
        if sorted_reminders:
            latest_reminder = sorted_reminders[0]
            task_id = latest_reminder[1]
            response = latest_reminder[3]
        else:
            # Handle the case where there are no reminders
            task_id = None  # or raise an exception, return a default value, etc.

        if task_id is not None:        
            if response == 'Not Done':
                c.execute("SELECT * FROM Tasks WHERE task_id = ?", (task_id,))
                task = c.fetchone()
                task_time = task[4]
                task_name = task[2]
                
                # convert str to datetime
                task_time = timezone.datetime.strptime(task_time, '%H:%M')
                
                print(type(task_time))
                task_time += timedelta(hours=2)
                print(type(task_time))
                task_time = task_time.strftime('%H:%M')
                if task[10] != 'Once':
                    if task[11] != None:
                        task_time = task[11]
                        task_time = timezone.datetime.strptime(task_time, '%H:%M')
                        task_time += timedelta(hours=2)
                        task_time = task_time.strftime('%H:%M')
                        
                    c.execute("UPDATE Tasks SET temp_time = ? WHERE task_id = ?", (task_time, task_id))
                    conn.commit()
                    print('task time', task_time,'updated')
                    c.execute("DELETE FROM \"Reminder Responses\" WHERE user_id = ? AND task_id = ?", (user_id, task_id))
                    conn.commit()
                    c.close()
                    msg = "Okay. I will remind you again to complete "+task_name+" at "+task_time+"."
                    client.messages.create(
                        from_='whatsapp:+917618207974',
                        body=msg,
                        to=user_number
                    )
                    return
                print(type(task_time))
                c.execute("UPDATE Tasks SET Completion_time = ? WHERE task_id = ?", (task_time, task_id))
                conn.commit()
                c.execute("DELETE FROM \"Reminder Responses\" WHERE user_id = ? AND task_id = ?", (user_id, task_id))
                conn.commit()
                c.close()
                msg = "Okay. I will remind you again to complete "+task_name+" at "+task_time+"."
                client.messages.create(
                    from_='whatsapp:+917618207974',
                    body=msg,
                    to=user_number
                )
                return
            c.execute("UPDATE Tasks SET status = ? WHERE task_id = ?", ('Complete', task_id))
            conn.commit()
            c.execute("DELETE FROM \"Reminder Responses\" WHERE user_id = ? AND task_id = ?", (user_id, task_id))
            conn.commit()
            c.close()
            client.messages.create(
                from_='whatsapp:+917618207974',
                body="Great! Keep it up!",
                to=user_number
            )
        
        else:

            client.messages.create(
                from_='whatsapp:+917618207974',
                body="Sorry, I didn't get that. Type 'help' to know more.",
                to=user_number
            )
    elif user_message == 'no' or user_message == 'not yet' or user_message == 'not done' or user_message == 'not completed' or user_message == 'not finished':
        c = conn.cursor()
        c.execute("SELECT * FROM Users WHERE mobile_number = ?", (user_number[12:],))
        user = c.fetchone()
        user_id = user[0]
        c.execute("SELECT * FROM \"Reminder Responses\" WHERE user_id = ?", (user_id,))
        reminders = c.fetchall()
        c.close()
        
        # Assuming index 4 represents the date and time in the format 'YYYY-MM-DD HH:MM:SS'
        # task_id = None
        sorted_reminders = sorted(reminders, key=lambda x: x[4], reverse=True)
        if sorted_reminders:
            latest_reminder = sorted_reminders[0]
            task_id = latest_reminder[1]
            response = latest_reminder[3]
        else:
            # Handle the case where there are no reminders
            task_id = None  # or raise an exception, return a default value, etc.

        if task_id is not None:
            if response == 'Not Done':
                c = conn.cursor()
                c.execute("UPDATE Tasks SET status = ? WHERE task_id = ?", ('Discarded', task_id))
                conn.commit()
                c.execute("DELETE FROM \"Reminder Responses\" WHERE user_id = ? AND task_id = ?", (user_id, task_id))
                conn.commit()
                c.close()
                client.messages.create(
                    from_='whatsapp:+917618207974',
                    body="Okay. I will Discard it.",
                    to=user_number
                )
                return
            c = conn.cursor()
            c.execute("UPDATE \"Reminder Responses\" SET responded = ? WHERE user_id = ? AND task_id = ?", ('Not Done',user_id, task_id))
            conn.commit()
            c.close()
            client.messages.create(
                from_='whatsapp:+917618207974',
                body="Will you be able to complete it today?",
                to=user_number
            )        
        else:
            client.messages.create(
                from_='whatsapp:+917618207974',
                body="Sorry, I didn't get that. Type 'help' to know more.",
                to=user_number
            )

    else:
        c = conn.cursor()
        c.execute("SELECT * FROM Users WHERE mobile_number = ?", (user_number[12:],))
        user = c.fetchone()
        user_id = user[0]
        c.execute("SELECT * FROM Tasks WHERE user_id = ? AND task_name = ?", (user_id, original_message))
        task = c.fetchone()
        if task is not None:
            task_id = task[0]
            task_name = task[2]
            task_time = task[4]
            task_status = task[8]
            if task_status == 'Incomplete':
                c.execute("UPDATE Tasks SET status = ? WHERE task_id = ?", ('Complete', task_id))
                conn.commit()
                c.close()
                client.messages.create(
                    from_='whatsapp:+917618207974',
                    body="Great! Keep it up! I marked this as complete.",
                    to=user_number
                )
            else:
                client.messages.create(
                    from_='whatsapp:+917618207974',
                    body="You have already completed this task.",
                    to=user_number
                )
        else:
            
            try:
                prompt = f"User: {original_message}\nChatGPT:"
                # response = openai.Completion.create(
                #     engine="text-davinci-002",  # Choose the appropriate engine
                #     prompt=prompt,
                #     max_tokens=50  # Adjust max_tokens as needed
                # )
                response = openai.chat.completions.create(
                    model="ft:gpt-3.5-turbo-0613:personal::8Xsif88y",  # Replace with your model identifier
                    messages=[
                        {"role": "system", "content": "You are TaskBlaze."},
                        {"role": "user", "content": prompt}
                    ]
                )

                chatgpt_response = response.choices[0].message.content
                print(chatgpt_response)

                # Send the message via WhatsApp
                client.messages.create(
                    from_='whatsapp:+917618207974',
                    body=chatgpt_response,
                    to=user_number
                )

            except Exception as e:
                print(f"An error occurred: {e}")
                # Here, you can handle the error as needed, 
                # for example, by sending a default message or logging the error.
                client.messages.create(
                    from_='whatsapp:+917618207974',
                    body="Sorry, I didn't get that. Type 'help' to know more.",
                    to=user_number
                )
    conn.close()


from datetime import datetime

@shared_task
def send_reminders_new():

    conn = sqlite3.connect('usertasks.db',check_same_thread=False)
    c = conn.cursor()
    todays_date = timezone.now().strftime('%d-%m-%Y')
    india_timezone = pytz.timezone('Asia/Kolkata')

    # Get the current time in India
    current_time = timezone.now().astimezone(india_timezone).strftime('%H:%M')
    print(todays_date, current_time)

    c.execute("SELECT * FROM Tasks WHERE status = ? AND date = ? AND Completion_time = ? AND Periodicity =?", ('Incomplete', todays_date, current_time, 'Once'))
    tasks = c.fetchall()
    for task in tasks:
        user_id = task[1]
        task_id = task[0]
        c.execute("SELECT * FROM Users WHERE User_Id = ?", (user_id,))
        user = c.fetchone()
        user_number = user[2]
        user_name = user[1]
        task_name = task[2]
        print(user_number, user_name, task_name, task_id, user_id)
        reminder_message = "Hey "+user_name+"! Have you completed: "+task_name + "? Reply with 'yes' or 'no'."
        user_number = 'whatsapp:+91'+user_number
        client.messages.create(
            from_='whatsapp:+917618207974',
            body=reminder_message,
            to=user_number
        )
        current_time = timezone.now()
        c.execute("INSERT INTO \"Reminder Responses\" (User_Id, task_id, responded, r_time) VALUES (?, ?, ?, ?)", (user_id, task_id, 'No', current_time))
        conn.commit()
        c.close()
    c = conn.cursor()
    c.execute("SELECT * FROM Tasks WHERE Periodicity != ?", ('Once',))
    tasks = c.fetchall()
    for task in tasks:
        if task[4] == current_time or task[11] == current_time:
            if task[4] == current_time:
                c.execute("UPDATE Tasks SET temp_time = ? WHERE task_id = ?", (None, task[0]))
                conn.commit()
            user_id = task[1]
            task_id = task[0]
            c.execute("SELECT * FROM Users WHERE User_Id = ?", (user_id,))
            user = c.fetchone()
            user_number = user[2]
            user_name = user[1]
            task_name = task[2]
            reminder_message = "Hey "+user_name+"! Have you completed: "+task_name + "? Reply with 'yes' or 'no'."
            user_number = 'whatsapp:+91'+user_number
            date_added = task[3]
            todays_date = timezone.now().strftime('%d-%m-%Y')
            todays_date = datetime.strptime(todays_date, '%d-%m-%Y').date()
            date_added = datetime.strptime(date_added, '%d-%m-%Y').date()
            difference = todays_date - date_added
            difference = difference.days
            if task[10] == 'Daily':
                client.messages.create(
                    from_='whatsapp:+917618207974',
                    body=reminder_message,
                    to=user_number
                )
                print('daily')
                c.execute("INSERT INTO \"Reminder Responses\" (User_Id, task_id, responded, r_time) VALUES (?, ?, ?, ?)", (user_id, task_id, 'No', current_time))
                conn.commit()
            elif task[10] == 'Weekly' and difference%7 == 0:
                client.messages.create(
                    from_='whatsapp:+917618207974',
                    body=reminder_message,
                    to=user_number
                )
                print('weekly')
                c.execute("INSERT INTO \"Reminder Responses\" (User_Id, task_id, responded, r_time) VALUES (?, ?, ?, ?)", (user_id, task_id, 'No', current_time))
                conn.commit()

            elif task[10] == 'Monthly' and difference%30 == 0:
                client.messages.create(
                    from_='whatsapp:+917618207974',
                    body=reminder_message,
                    to=user_number
                )
                print('monthly')
                c.execute("INSERT INTO \"Reminder Responses\" (User_Id, task_id, responded, r_time) VALUES (?, ?, ?, ?)", (user_id, task_id, 'No', current_time))
                conn.commit()
    c.close()
    conn.close()


@shared_task
def reminder_to_registered_user_every_morning_new():
    conn = sqlite3.connect('usertasks.db',check_same_thread=False)
    reminder_time = '07:00'
    # reminder_time = '19:28'
    india_timezone = pytz.timezone('Asia/Kolkata')

    # Get the current time in India
    current_time = timezone.now().astimezone(india_timezone).strftime('%H:%M')
    print(current_time)
    if reminder_time == current_time:
        conn = sqlite3.connect('usertasks.db',check_same_thread=False)
        c = conn.cursor()

        c.execute("SELECT * FROM Users")
        users = c.fetchall()
        c.execute("DELETE FROM \"Reminder Responses\"");
        conn.commit()

        for user in users:
            user_id = user[0]
            # print(user_id)
            user_number = user[2]
            user_name = user[1]
            reminder_message = "Hey "+user_name+"\n\n! Good Morning! \n\n Add your tasks for today here:\n\n http://taskblaze.tech/add_task/"
            reminder_message += user_number
            user_number = 'whatsapp:+91'+user_number
            print('test')
            client.messages.create(
                from_='whatsapp:+917618207974',
                body=reminder_message,
                to=user_number
            )
            current_time = timezone.now()
            
        c.close()
        conn.close()


# reminder_to_registered_user_every_morning_new()

# send_reminders_new()


# def testing_gpt():
#     original_message = "Hello i feel less energetic today"
#     prompt = f"User: {original_message}\nChatGPT:"
#     response = openai.Completion.create(
#         engine="text-davinci-002",  # Choose the appropriate engine
#         prompt=prompt,
#         max_tokens=50  # You can adjust the max_tokens based on the desired length of the response
#     )
    
#     chatgpt_response = response.choices[0].text.strip()
#     print(chatgpt_response)

# testing_gpt()
        
