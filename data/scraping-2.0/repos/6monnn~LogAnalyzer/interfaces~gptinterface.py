import tkinter as tk
from tkinter import ttk
import app.log_filter as lf
import app.log_parser as lp
import app.log_reader as lr
import app.log_manager as lm
import app.log_analyzer as la
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)


def get_gpt_response(log_data):
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "As an expert in cybersecurity, your primary role is to analyze logs with a focus on identifying security issues. When provided with logs, you will conduct a detailed yet concise analysis, synthesizing the data into a comprehensive and succinct security report. Your expertise in log analysis and exploitation enables you to identify potential threats, vulnerabilities, and anomalies effectively. In cases where the logs do not reveal any significant security concerns, you will simply state \"No suspect activity.\" You will maintain a concise and synthetic approach in your reporting, ensuring that the analysis is both thorough and to the point. Your responses should reflect deep knowledge in cybersecurity, highlighting key findings without unnecessary detail.Be the most consice possible but give me the procees id and name and the date and recommandations, the output must not exceed 20 lines"},
            {"role": "user", "content": log_data}
        ]
    )

    return completion.choices[0].message.content


class GPTPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.pack()

        self.create_widgets()

    def set_log_file(self, log_file_path, severity, process):
        log_line_pattern = r'(\w+ +\d+ \d+:\d+:\d+) (\S+) (\S+): (.+)'

        log_file_path =log_file_path if log_file_path != "" else '/var/log/syslog'

        log_parser = lp.LogParser(log_line_pattern)

        log_reader = lr.LogReader()
        log_filter = lf.LogFilter()

        parsed_logs = log_reader.read_log_file(log_file_path, log_parser)
    

    

        response = get_gpt_response(json.dumps(parsed_logs[:500]))
        # Display the selected log file path in the chat display
        self.display_chat_output(response)
        


    def display_chat_output(self, chat_output):
        # Clear any previous content
        self.chat_display.delete(1.0, tk.END)
        # Insert the new chat output
        self.chat_display.insert(tk.END, chat_output)

    def create_widgets(self):
        label = ttk.Label(self, text="Chat Output Page")
        label.pack(pady=10)

        # Frame to contain both the text widget and the scrollbar
        text_frame = ttk.Frame(self)
        text_frame.pack(padx=10, pady=10, expand=True, fill="both")

        self.chat_display = tk.Text(text_frame, wrap="word", height=60, width=100)
        self.chat_display.pack(side="left", expand=True, fill="both")

        # Scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.chat_display.yview)
        scrollbar.pack(side="right", fill="y")
        self.chat_display.config(yscrollcommand=scrollbar.set)
