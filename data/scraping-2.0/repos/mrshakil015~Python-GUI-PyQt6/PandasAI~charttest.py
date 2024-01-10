# OPENAI_API_KEY = "k-mtujzCGAyXF5x2gLcLBDT3BlbkFJVLdrDwDxsKnpEkhzV7t"
import sys
import csv
import pandas as pd
import matplotlib  # Add this import
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
# from io import BytesIO
# from PIL import Image
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTextBrowser, QTextEdit, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QLabel

from PyQt6.QtCore import Qt
import os
from PyQt6.QtGui import QPixmap

# matplotlib.use('Agg')  # Move this line here



class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Chat Window")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.chat_display_label = QLabel()
        self.layout.addWidget(self.chat_display_label)

        self.message_input = QTextEdit()
        self.message_input.setFixedHeight(50)
        self.layout.addWidget(self.message_input)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.layout.addWidget(self.send_button)

        self.upload_button = QPushButton("Upload CSV")
        self.upload_button.clicked.connect(self.upload_csv)
        self.layout.addWidget(self.upload_button)

        self.table_widget = QTableWidget()
        self.layout.addWidget(self.table_widget)

        self.central_widget.setLayout(self.layout)

        self.chat_history = []
        self.df = None  # Store the CSV data

        self.init_pandasai()

    def init_pandasai(self):
        OPENAI_API_KEY = "k-mtujzCGAyXF5x2gLcLBDT3BlbkFJVLdrDwDxsKnpEkhzV7t"
        llm = OpenAI(api_token=OPENAI_API_KEY)
        user_defined_path = os.getcwd()
        pandas_ai = PandasAI(llm, verbose=True, conversational=False, save_charts=True,
                            save_charts_path=user_defined_path, enable_cache=False)

        self.pandas_ai = PandasAI(llm)

    # def send_message(self):
    #     message = self.message_input.toPlainText()
    #     if message:
    #         self.chat_history.append(f"User: {message}")
    #         response = self.run_pandasai(message)
    #         if response is not None:  
    #             if isinstance(response, float):
    #                 self.chat_history.append(f"PandasAI: {response}")
    #             elif isinstance(response, str) and "plot" in response.lower():
    #                 self.generate_and_display_plot(response)
    #             else:
    #                 self.chat_history.append(f"PandasAI: {response}")
    #             self.update_chat_display()

    #         print(response)

    def send_message(self):
        message = self.message_input.toPlainText()
        if message:
            self.chat_history.append(f"User: {message}")
            response = self.run_pandasai(message)
            if response is not None:
                if isinstance(response, float):
                    self.chat_history.append(f"PandasAI: {response}")
                elif isinstance(response, str) and "plot" in response.lower():
                    self.generate_and_display_plot(response)
                else:
                    self.chat_history.append(f"PandasAI: {response}")
                self.update_chat_display()

            print(response)

    def run_pandasai(self, message):
        try:
            if self.df is not None:
                
                response = self.pandas_ai.run(self.df, prompt=message)
                plt.show()
                # code = self.pandas_ai.last_code_generated(self.df, prompt=message)
                print("Pandasai response:", response)
                return response
            else:
                return "Please upload a CSV file first."
        except Exception as e:
            print("PandasAI Error:", e)
            return None

    def update_chat_display(self):
        chat_text = "\n".join(self.chat_history)
        self.chat_display_label.setText(chat_text)


    def upload_csv(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_name, _ = file_dialog.getOpenFileName(self, "Upload CSV File", "", "CSV Files (*.csv);;All Files (*)")

        if file_name:
            try:
                self.df = pd.read_csv(file_name)
                self.load_csv_to_table(file_name)
                self.chat_history.append("CSV file uploaded.")
                self.update_chat_display()
            except Exception as e:
                print("Error:", e)

    def load_csv_to_table(self, file_name):
        self.table_widget.clear()
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(0)

        with open(file_name, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row_idx, row in enumerate(csv_reader):
                if row_idx == 0:
                    self.table_widget.setColumnCount(len(row))
                self.table_widget.insertRow(row_idx)
                for col_idx, cell_value in enumerate(row):
                    self.table_widget.setItem(row_idx, col_idx, QTableWidgetItem(cell_value))


    # def generate_and_display_chart(self, data):
    #     plt.figure()
    #     plt.title("Generated Chart")
    #     plt.bar(range(len(data)), data)
    #     plt.xlabel("X-axis")
    #     plt.ylabel("Y-axis")
    #     plt.tight_layout()

    #     # Display the figure using Matplotlib and process PyQt events
    #     plt.show(block=True)
    #     QApplication.processEvents()

    #     self.chat_history.append("PandasAI: Generated Chart")


    def clear_layout(self):
        layout = self.layout()
        if layout.count() > 5:
            for i in range(5, layout.count()):
                widget = layout.itemAt(i).widget()
                if widget:
                    widget.deleteLater()

def main():
    app = QApplication(sys.argv)
    chat_app = ChatWindow()
    chat_app.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
