import os
import time
import base64
import json
import math
import re
import subprocess
import pyautogui
import argparse
import platform
import datetime
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageGrab
import Xlib.display
import Xlib.X
import Xlib.Xutil 
from openai import OpenAI
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame
from PyQt5.QtCore import Qt

load_dotenv()

client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

monitor_size = {
    "width": 1920,
    "height": 1080,
}

ASSISTANT_PROMPT = """
You are a Self-Operating Computer and Quiz Assistant.

View the screenshot identify all present quiz question(s), and answers if applicable. Ignore the window titled Sightseer-GPT4V.

Answer the question(s) and provide assistance to finding the answer. Do not provide the answer. Explain your reasoning.

Answer in the format for each question:
QUESTION: <question>
ASSISTANT: <answer>

If there are multiple questions, help with each of them. Be direct and straightforward in reasoning. If the question is simple, explain in 1-2 sentences. If it is complex or multi-step reasoning, use 3-4 sentences.

ONLY return the formatted response, nothing else. If you cannot process the image or are unable to answer the question, type "Unable to answer" and move on to the next question.

"""

def screenshot_name():
    # Get the current timestamp
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define the base file name
    base_file_name = "screenshots/screenshot"

    # Combine the base file name with the formatted timestamp and file extension
    file_path = f"{base_file_name}_{current_timestamp}.png"

    return file_path

def capture_screen_with_cursor(file_path):
    user_platform = platform.system()
    if user_platform == "Windows":

        screenshot = pyautogui.screenshot()
        screenshot.save(file_path)
    elif user_platform == "Linux":
        # Use xlib to prevent scrot dependency for Linux
        screen = Xlib.display.Display().screen()
        size = screen.width_in_pixels, screen.height_in_pixels
        monitor_size["width"] = size[0]
        monitor_size["height"] = size[1]
        screenshot = ImageGrab.grab(bbox=(0, 0, size[0], size[1]))
        screenshot.save(file_path)
    elif user_platform == "Darwin":  # (Mac OS)
        # Use the screencapture utility to capture the screen with the cursor
        subprocess.run(["screencapture", "-C", file_path])
    else:
        print(f"The platform you're using ({user_platform}) is not currently supported")

def ask_assistant(file_path):
    try:
        with open(file_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": ASSISTANT_PROMPT},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                    },
                    },
                ],
                }
            ],
            max_tokens=1200,
        )

        content = response.choices[0].message.content
        return content
    
    except Exception as e:
        print(f"Error: {e}")
        return (f"Error: {e}")

def workflow():
    file_path = screenshot_name()
    capture_screen_with_cursor(file_path)
    response = ask_assistant(file_path)
    return response

class ApplicationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.workflow = workflow
        self.initUI()

    def run_workflow_button(self):
        # Call the external function and update the label
        response = self.workflow()
        self.label.setText(response)

    def initUI(self):
        self.setWindowTitle("Sightseer-GPT4V")
        self.setGeometry(0, 0, 350, 950)  # Set the geometry (x, y, width, height)

        # Create a vertical layout
        layout = QVBoxLayout()

        # Create a label with some text
        self.label = QLabel("welcome to sightseer")
        self.label.setWordWrap(True)  # Enable word wrap

        # Create a scroll area, add the label to it, and set the widget resize policy
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allow the widget to be resized
        scroll_area.setWidget(self.label)  # Set the label as the scroll area's widget
        scroll_area.setFrameShape(QFrame.NoFrame)  # Set frame shape to NoFrame to remove the border

        # Add the scroll area to the layout instead of the label
        layout.addWidget(scroll_area)
        self.label.setAlignment(Qt.AlignTop)  # Align text to the top

        # Create a button that calls run_workflow_button when clicked
        button = QPushButton("Run")
        button.clicked.connect(self.run_workflow_button)
        layout.addWidget(button)

        # Set the layout on the main window
        self.setLayout(layout)

# Create the application object
app = QApplication([])

# Create the main window
window = ApplicationWindow()

# Show the window
window.show()

# Run the application
app.exec_()