import tkinter as tk
from tkinter import ttk, scrolledtext
import openai
import json


class InsightsPage:
    def __init__(self, root, data):
        self.root = root
        self.data = data
        self.root.title("Insights")
        self.root.geometry("500x600")

        insights_text = self.fetch_insights()
        
        scrolled_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=40, height=5, font=("Arial", 14))
        scrolled_text.insert(tk.END, insights_text)
        scrolled_text.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        exit_button = ttk.Button(self.root, text="Exit", command=lambda: self.exit_program(self.root))
        exit_button.pack(pady=20)

    def fetch_insights(self):
        data=self.data
        openai.api_key = ""# Replace with your actual OpenAI API key
        data_json = json.dumps(data)
        user_message = {"role": "user", "content": data_json}
        messages = [
            {"role": "system", "content": f"""
            Given a user's workout data in JSON format, including their name, exercise duration, number of reps, and errors, generate personalized exercise tips and only one motivation quote.            
            """},
            {"role": "user", "content": data_json}
        ]
        response = self.get_completion(messages)
        return response

    def get_completion(self, messages, model="gpt-3.5-turbo"):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.9,  # this is the degree of randomness of the model's output 0-2
        )
        return response.choices[0].message["content"]

    def exit_program(self, window=None):
        if window:
            window.destroy()
        else:
            self.root.destroy()
