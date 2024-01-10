from openai import OpenAI
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QFileDialog, \
    QCheckBox, QLineEdit, QLabel
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
import sys
import base64

href = "https://openai.com/blog/new-models-and-developer-products-announced-at-devday"


class SettingsWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Conciseness checkbox
        self.compact_checkbox = QCheckBox("Conciseness")
        self.compact_checkbox.setChecked(True)  # Checked by default

        # Only Hungarian language checkbox
        self.hungarian_checkbox = QCheckBox("Only Hungarian language")
        self.hungarian_checkbox.setChecked(True)  # Checked by default

        # Custom prompt text
        self.prompt_line_edit = QLineEdit()
        self.prompt_line_edit.setText("You are a professional and kind university teacher. An expert in university subjects!")
        self.prompt_line_edit.setPlaceholderText("Custom prompt text")

        # label
        self.prompt_label = QLabel("Custom ChatGPT Prompt:")
        self.about = QLabel(
            f"This application uses the gpt-4-vision model <a href=\"{href}\">(Here)</a>. Not the free "
            f"version(3.5). Please, do not share. :)")
        self.about.setOpenExternalLinks(True)

        # Layout and creation of widgets
        layout = QVBoxLayout()
        layout.addWidget(self.compact_checkbox)
        layout.addWidget(self.hungarian_checkbox)
        layout.addWidget(self.prompt_label)
        layout.addWidget(self.prompt_line_edit)
        layout.addWidget(self.about)
        self.setLayout(layout)


class ChatGPTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IGPT (ChatGPT 4.0 Vision API Chat App)")
        self.resize(800, 600)

        # Layout and creation of widgets
        layout = QVBoxLayout()

        self.text_edit_input = QTextEdit()
        self.text_edit_output = QTextEdit()
        self.text_edit_input.setPlaceholderText("Enter your question here. You can also attach an image! Optionally, specify "
                                                "what kind of image you want to generate with the DALL-E 3 model.")
        self.send_button = QPushButton("Send")
        self.select_image_button = QPushButton("Select Image")
        self.generate_image_button = QPushButton("Generate Image (DALL-E-3)")
        layout.addWidget(self.text_edit_input)
        layout.addWidget(self.select_image_button)
        layout.addWidget(self.generate_image_button)
        layout.addWidget(self.text_edit_output)
        layout.addWidget(self.send_button)

        # Add settings widget to the bottom of the window
        self.settings_widget = SettingsWidget()
        layout.addWidget(self.settings_widget)

        # Signal/slot connection
        self.send_button.clicked.connect(self.send_message)
        self.select_image_button.clicked.connect(self.select_image)
        self.generate_image_button.clicked.connect(self.generate_image)

        # Set the widget as the central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Initialize OpenAI with the API key
        self.openai_client = OpenAI(api_key="YOUR-API-KEY")

        # Path to the selected image
        self.selected_image_path = None

    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setOptions(options)

        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.selected_image_path = selected_files[0]
                print(f"Selected image: {self.selected_image_path}")

    def send_message(self):
        # Get input text or image
        input_text = self.text_edit_input.toPlainText()

        # API call without an image if no image is selected
        if not self.selected_image_path:
            response = self.chat_with_gpt(input_text)
        else:
            # API call with an image if an image is selected
            response = self.chat_with_gpt_and_image(input_text, self.selected_image_path)

        # Display the output
        if response:
            self.text_edit_output.append(f"<font color='blue'>GPT-4 Ivett: </font>" + response)

    def chat_with_gpt(self, text):
        compact = "Express concisely." if self.settings_widget.compact_checkbox.isChecked() else ""
        hungarian = "Answer in Hungarian." if self.settings_widget.hungarian_checkbox.isChecked() else ""
        prompt = self.settings_widget.prompt_line_edit.text()

        # API call without an image
        response = self.openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"{text}\n{compact}, {hungarian}\nPrompt: {prompt}"}],
                }
            ],
            max_tokens=300,
        )

        # Process the response
        if response.choices:
            return response.choices[0].message.content
        else:
            print("No choices found in the response.")
            return None

    def chat_with_gpt_and_image(self, text, image_path):
        compact = "Compact" if self.settings_widget.compact_checkbox.isChecked() else "No Compact"
        hungarian = "Hungarian" if self.settings_widget.hungarian_checkbox.isChecked() else "No Hungarian"
        prompt = self.settings_widget.prompt_line_edit.text()

        # Read and base64 encode the image file
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # API call with an image
        response = self.openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": f"{text}\nCompact: {compact}, Hungarian: {hungarian}\nPrompt: {prompt}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ],
                }
            ],
            max_tokens=500,
        )

        # Process the response
        if response.choices:
            return response.choices[0].message.content
        else:
            print("No choices found in the response.")
            return None

    def generate_image(self):
        # Get input text
        input_text = self.text_edit_input.toPlainText()

        # API call using the DALL-E-3 model
        response = self.openai_client.images.generate(
            model="dall-e-3",
            prompt=input_text,  # Generate an image based on the text provided by the user
            size="1024x1024",  # Image size
            quality="standard",  # Image quality
            n=1,  # Number of generated images
        )

        # URL of the first generated image
        image_url = response.data[0].url
        print(f"Generated image URL: {image_url}")

        # Open the image URL in the browser
        QDesktopServices.openUrl(QUrl(image_url))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatGPTApp()
    window.show()
    sys.exit(app.exec())
