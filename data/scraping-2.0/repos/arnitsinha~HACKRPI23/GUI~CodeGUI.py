import os
import openai
from rpikeys.settings import set_openai_key, set_google_calendar_credentials
from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
import threading


# Replace with your OpenAI API key
api_key = set_openai_key()

def chat_with_gpt(input_text):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=input_text,
        max_tokens=500,
        api_key=api_key
    )

    return response.choices[0].text.strip()

def create_google_calendar_event(title, description, start_time, end_time, calendar_id, credentials, timezone):
    service = build('calendar', 'v3', credentials=credentials)

    event = {
        'summary': title,
        'description': description,
        'start': {
            'dateTime': start_time,
            'timeZone': timezone,
        },
        'end': {
            'dateTime': end_time,
            'timeZone': timezone,
        },
    }

    event = service.events().insert(calendarId=calendar_id, body=event).execute()
    
    event_id = event['id']
    event_link = f'https://calendar.google.com/calendar/r/event?eid={event_id}'
    
    return event_link

def create_and_share_public_calendar(credentials, learn, level, days, timezone):
    service = build('calendar', 'v3', credentials=credentials)
    
    calendar = {
        'summary': f'Personalized Schedule for Learning {learn}',
        'description': f'This calendar was created by a bot. It contains a personalized schedule for learning {learn} for {level} in {days} days.',
        'timeZone': timezone,
    }
    
    created_calendar = service.calendars().insert(body=calendar).execute()
    calendar_id = created_calendar['id']
    
    rule = {
        'scope': {
            'type': 'default',
        },
        'role': 'reader',
    }
    service.acl().insert(calendarId=calendar_id, body=rule).execute()
    
    return calendar_id

class ScheduleApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Schedule Creator")
        self.setGeometry(100, 100, 400, 250)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        self.learn_input = QLineEdit(self)
        self.level_input = QLineEdit(self)
        self.days_input = QLineEdit(self)
        self.timezone_input = QLineEdit(self)

        layout.addWidget(QLabel("What do you want to learn? (Use this format: Python): "))
        layout.addWidget(self.learn_input)
        layout.addWidget(QLabel("What is your level? (Beginner, Intermediate, Advanced): "))
        layout.addWidget(self.level_input)
        layout.addWidget(QLabel("How many days do you want to learn this skill in? (Use this format: 3): "))
        layout.addWidget(self.days_input)
        layout.addWidget(QLabel("What is your timezone? (Use this format: EST, PST, etc.): "))
        layout.addWidget(self.timezone_input)

        self.schedule_button = QPushButton("Import Schedule File", self)
        self.schedule_button.clicked.connect(self.create_schedule)

        layout.addWidget(self.schedule_button)

        self.print_link_button = QPushButton("Calendar Link", self)
        self.print_link_button.clicked.connect(self.open_last_print_link)
        self.print_link_button.setEnabled(False)  # Initially disabled

        layout.addWidget(self.print_link_button)

        self.central_widget.setLayout(layout)

        self.last_print_link = None

    def create_schedule(self):
        learn = self.learn_input.text()
        level = self.level_input.text()
        days = self.days_input.text()
        timezone = self.timezone_input.text()
        credentials = service_account.Credentials.from_service_account_file(set_google_calendar_credentials())


        user_input = f"Make me a {days} day schedule for learning {learn} for {level}. Do not combine 2 days together. Give me just the days and the title together and then the task (Only One Task a Day.) to be performed that day as a single sentence. Also, include helpful URLs in the task sentence. Precede day with a # and the tasks with a *."

        response = chat_with_gpt(user_input)

        titles = []
        tasks = []

        schedule_text = response

        lines = schedule_text.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                titles.append(line)
            elif line.startswith('*'):
                tasks.append(line)

        start_date = datetime.now()
        calendar_id = create_and_share_public_calendar(credentials, learn, level, days, timezone)

        for i in range(len(titles)):
            title = titles[i].lstrip('#').strip()
            task = tasks[i].lstrip('*').strip()

            start_time = start_date.strftime('%Y-%m-%dT00:00:00')
            end_date = start_date + timedelta(days=1)
            end_time = end_date.strftime('%Y-%m-%dT23:59:59')

            event_link = create_google_calendar_event(title, task, start_time, end_time, calendar_id, credentials, timezone)

            start_date = end_date

        self.last_print_link = f'https://calendar.google.com/calendar/r?cid={calendar_id}'
        self.print_link_button.setEnabled(True)  # Enable the button

    def open_last_print_link(self):
        if self.last_print_link:
            # Open the last print link in the default web browser
            QDesktopServices.openUrl(QUrl(self.last_print_link))

if __name__ == "__main__":
    app = QApplication([])
    window = ScheduleApp()
    window.show()
    app.exec()