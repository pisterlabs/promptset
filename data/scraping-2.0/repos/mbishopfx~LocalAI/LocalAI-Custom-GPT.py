import os
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI 
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from PyQt5.QtWidgets import QFileDialog

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "api key here"
openai.api_key = "api key here"

# Enable to save to disk & reuse the model (STAY FALSE)
PERSIST = False

# Initialize the OpenAI model and index (similar to your previous script)
loader = DirectoryLoader("data/") # Copy the path and paste it using double slash's //
if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
else:
    index = VectorstoreIndexCreator().from_loaders([loader])

# Define a variable to store the user's input and conversation
user_input = ""
conversation = []

# Maximum tokens allowed
max_tokens = 7600  # Adjust as needed

# Define a function to query the OpenAI model for refining and debugging code
def query_openai_model(input_text, engine):
    global user_input, conversation

    # Define your custom system message here
    system_message = "Please provide information about the task you want me to perform."

    prompt = input_text

    # Calculate the total tokens used in the conversation
    total_tokens = openai.Tokenizer(conversation).num_tokens

    # Ensure we stay within the token limit
    while total_tokens > max_tokens:
        removed_tokens = openai.Tokenizer(conversation[0]['content']).num_tokens
        total_tokens -= removed_tokens
        conversation.pop(0)

    # Add the system message to the conversation
    conversation.append({"role": "system", "content": system_message})

    # Append the user's input to the conversation
    conversation.append({"role": "user", "content": input_text})

    response = openai.ChatCompletion.create(
        model=engine,
        messages=conversation,
        max_tokens=max_tokens,
    )

    response_text = response['choices'][0]['message']['content']

    user_input = input_text
    conversation.append({"role": "assistant", "content": response_text})

    return response_text

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 15}),
)

# PyForms GUI Integration
import sys
from pyforms import BaseWidget
from pyforms.controls import (
    ControlTextArea,  # Use ControlTextArea for multiline input
    ControlButton,
    ControlTextArea,
    ControlLabel,
)
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
class ConversationalRetrievalApp(BaseWidget):

    def __init__(self):
        super().__init__('LocalAI')

        self._instructions = ControlLabel('Welcome to LocalAI. You can interact with your data by prompting a question below.')

        self._query_input = ControlTextArea('Enter your query:')  
        self._search_button = ControlButton('Generate')
        self._response_display = ControlTextArea('Response:')

        self._search_button.value = self._on_search_button_click

        # Adjust the width and height of the query input
        self._query_input.width = 400  # Change the width as needed
        self._query_input.height = 100  # Set the height to double

        self.formset = [
            '_instructions',
            ('_query_input', '_search_button'),
            '_response_display'
        ]

        self._instructions.font = '24px'
        self._query_input.font = '20px'
        self._search_button.font = '20px'
        self._response_display.font = '18px'

    def _on_search_button_click(self):
        query = self._query_input.value
        if not query:
            return

        result = chain({"question": query, "chat_history": []})
        self._response_display.value = result['answer']
        from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication([])
    app.setStyleSheet('''
        QWidget {
            background-color: #333;
            color: #fff;
        }
        QTextEdit {
            background-color: #444;
            color: #fff;
            border: 1px solid #555;
        }
        QPushButton {
            background-color: #555;
            color: #fff;
        }
    ''')
    
    window = ConversationalRetrievalApp()
    window.show()
    sys.exit(app.exec_())
