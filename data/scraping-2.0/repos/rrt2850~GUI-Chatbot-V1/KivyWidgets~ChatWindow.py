"""
Author: Robert Tetreault (rrt2850)
Filename: ChatWindow.py
Description: This script contains the constructors and functions used by the chat window.
             this is makes a kivy widget responsible for displaying the chat window and
             handling the user input.
"""
import json
import os
import re

        
import dotenv
import openai
import tiktoken


os.environ["KIVY_NO_CONSOLELOG"] = "1"
os.environ['KIVY_TEXT'] = 'pil'

from kivy.uix.modalview import ModalView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput

from CharacterScripts.DataHandler import sharedVars

# set up environment variables
dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def countTokens(text):
    """
    counts the tokens in a message and returns the count
    """
    tokens = []
    try:
        tokens = encoding.encode(text)
    except:
        print("Error: Unable to encode text")
    return len(tokens)

class CustomTextInput(TextInput):
    """
    custom text input that allows for the enter key to be used to send messages
    """
    def _keyboard_on_key_down(self, window, keycode, text, modifiers):
        if keycode[1] == "enter" and not modifiers:
            self.dispatch("on_text_validate")
            self.focus = True
            return True
        return super()._keyboard_on_key_down(window, keycode, text, modifiers)
    

class ChatBoxLayout(BoxLayout):
    """
    A box layout that contains a scrollable chat window and an input box
    """

    def __init__(self, **kwargs):
        super(ChatBoxLayout, self).__init__(**kwargs)

        self.orientation = 'vertical'
        self.padding = 10
        self.spacing = 4
        self.totalTotal = 0
        self.updateVars()

        self.setupUIComponents()

        self.loadMessages()

    def setupUIComponents(self):
        """
        Makes the scroll view and input box and adds them to the layout
        """
        self.setupScrollView()
        self.setupInputLayout()
        self.add_widget(self.scrollView)
        self.add_widget(self.inputLayout)

    def setupScrollView(self):
        self.scrollView = self.createScrollView()
        self.messageLabel = self.createMessageLabel()
        self.scrollView.add_widget(self.messageLabel)

    def createScrollView(self):
        """
        Makes a scroll view to hold the messages so that they can be scrolled through if they don't fit on the screen
        """
        return ScrollView(
            bar_width=10,
            effect_cls='ScrollEffect',
            scroll_type=['bars', 'content'],
            scroll_wheel_distance=10
        )

    def createMessageLabel(self):
        """
        Makes a label to hold the messages
        """
        messageLabel = Label(
            size_hint_y=None,
            size_hint_x=0.9,
            text_size=(self.width, None),
            font_size='16sp',
            markup=True,
            valign='top',
            halign='left',
            padding=(10, 10)
        )
        messageLabel.bind(
            width=lambda *x: messageLabel.setter('text_size')(messageLabel, (messageLabel.width, None)),
            texture_size=lambda *x: messageLabel.setter('height')(messageLabel, messageLabel.texture_size[1])
        )
        return messageLabel

    def setupInputLayout(self):
        """
        Makes a layout to hold the input box and send button
        """
        self.inputLayout = self.createInputLayout()
        self.userInput = self.createUserInput()
        self.sendButton = self.createSendButton()

        self.inputLayout.add_widget(self.userInput)
        self.inputLayout.add_widget(self.sendButton)

    def createInputLayout(self):
        """
        Makes a layout to hold the input box and send button
        """
        return BoxLayout(
            size_hint_y=0.1,
            orientation='horizontal'
        )

    def createUserInput(self):
        """
        Makes a text input box for the user to type in
        """

        userInput = CustomTextInput(
            multiline=False,
            do_wrap=True,
            hint_text='Type your message...',
            font_size='16sp',
            size_hint_x=0.9,
            text_validate_unfocus=False
        )
        userInput.bind(on_text_validate=lambda instance: self.sendMessage())
        return userInput

    def createSendButton(self):
        """
        Makes a button to send the message
        """

        sendButton = Button(
            text='Send',
            font_size='16sp',
            size_hint_x=0.1
        )
        sendButton.bind(on_press=self.sendMessage)
        return sendButton

    def updateVars(self):
        temp = sharedVars.gptStuff
        self.temperature = temp["temperature"]
        self.topP = temp["topP"]
        self.maxTokens = temp["maxTokens"]
        self.frequencyPenalty = temp["frequencyPenalty"]
        self.presencePenalty = temp["presencePenalty"]
        self.tokenLimit = temp["tokenLimit"]
        self.prompt = sharedVars.prompt
        self.systemMessage = sharedVars.setting

    def loadMessages(self):
        """
        loads message history from json file to resume previous chat
        """
        # Attempt to load the chat history from the JSON file
        try:
            self.chatHistory = self.loadChatHistoryFromFile()
            messages = self.chatHistory["logs"].get(sharedVars.chatKey, [])
        except (FileNotFoundError, json.JSONDecodeError):  # If file not found or there's an error decoding JSON
            # Start a new chat if chat history can't be loaded
            self.startNewChat()
            return

        # If no messages were found in the chat history
        if not messages:
            # Start a new chat if there are no messages in the chat history
            self.startNewChat()
        else:
            # Otherwise, proceed with the existing chat history
            sharedVars.messages = messages
            self.handleExistingChat()


    def loadChatHistoryFromFile(self):
        # Load and return the chat history from the JSON file
        with open(f"CharacterJsons/ChatHistory{sharedVars.player.name}.json", 'r') as f:
            return json.load(f)


    def startNewChat(self):
        # Define the initial messages for a new chat
        messages = [
            {"role": "user", "content": self.prompt},
            {"role": "system", "content": self.systemMessage}
        ]

        # Add each initial message to the GUI and save it in the chat history
        self.saveChatHistory(messages[0])
        self.saveChatHistory(messages[1])
        self.appendMessage(messages[1])
        

        # Update the global messages variable and scroll to the bottom
        sharedVars.messages = messages
        self.scrollView.scroll_y = 0

        # Start the chat loop
        self.chatLoop()

    def handleExistingChat(self):
        self.keepLoading = False

        def yes(button):
            self.keepLoading = True
            modal_view.dismiss()

        def no(button):
            self.keepLoading = False
            modal_view.dismiss()

        box = BoxLayout(orientation='vertical')
        box.add_widget(Label(text='Load previous chat history?'))
        box.add_widget(Button(text='Yes', on_release=yes))
        box.add_widget(Button(text='No', on_release=no))

        modal_view = ModalView(size_hint=(0.5, 0.5), auto_dismiss=False)
        modal_view.add_widget(box)
        modal_view.bind(on_dismiss=self.on_dismiss_popup)
        modal_view.open()

    def on_dismiss_popup(self, instance):
        if not self.keepLoading:
            # clear existing chat history
            self.chatHistory["logs"][sharedVars.chatKey] = []
            with open(f"CharacterJsons/ChatHistory{sharedVars.player.name}.json", 'w') as f:
                json.dump(self.chatHistory, f, indent=4)

            self.startNewChat()
            return

        messages = sharedVars.messages
        
        # Replace the first message in the history with the current prompt
        messages[0] = {"role": "system", "content": self.prompt}

        # Load all the messages into the GUI message holder
        for message in messages[1:]:
            self.appendMessage(message)

        # Scroll to the bottom
        self.scrollView.scroll_y = 0

        # If a new system message is found, add it to the chat history, taking priority over user messages
        if messages[1] != {"role": "system", "content": self.systemMessage}:
            messages.append({"role": "system", "content": self.systemMessage})

        # Update global messages variable
        sharedVars.messages = messages

        # If the last message is not a response, respond to it
        if messages[-1]["role"] != "assistant":
            self.chatLoop()

    def appendMessage(self, message):
        """
        formats and adds a message to the GUI
        """

        # initialize the colors for each role and the formatted message
        roleColors = {"system": "#ADD8E6", "user": "#32CD32", "assistant": "#800080", "character": "#800080", "character2": "#8B4513"}
        formattedMessage = ""

        # set the role and get the name based on the role
        role = message["role"]
        name = "System" if role == "system" else "You" if role == "user" else sharedVars.currCharacter.name.first

        # initialize the character names
        character1 = sharedVars.currCharacter.name.first
        character2 = sharedVars.currCharacter2.name.first if sharedVars.currCharacter2 else ""

        # if the message is a response from the chatbot
        if role == "assistant":
            if character2:
                # parse the message into dialogues
                pattern = f'({character1}|{character2}): (.*?)(?=(?:{character1}|{character2}):|$)'
                dialogues = re.findall(pattern, message['content'], re.DOTALL)

                # if dialogues were found, format them accordingly
                if dialogues:
                    for speaker, text in dialogues:
                        # get the role of the speaker
                        speakerRole = 'character' if speaker == character1 else 'character2'

                        # Color code for the speaker role
                        colorCode = roleColors[speakerRole]

                        # Clean up the text
                        cleanedText = re.sub('\n+', '\n', text)

                        # Create the formatted string
                        formattedMessage += f"\n\n[color={colorCode}][b]{speaker}:[/b][/color] {cleanedText}"
                else:
                    role = 'system'
                    name = 'System?'
            else:
                formattedMessage = re.sub(f'{character1}: ', '', message['content'])
                # if there is only one character, format the message accordingly
                formattedMessage = f"\n\n[color={roleColors[role]}][b]{name}:[/b][/color] {formattedMessage}"
                

        # if the message is a system message or a user message, format it accordingly
        if not formattedMessage:
            # format the message
            message["content"] = re.sub('\n+', '\n', message["content"])
            formattedMessage = f"\n\n[color={roleColors[role]}][b]{name}:[/b][/color] {message['content']}"

        # remove all double quotes from the message
        formattedMessage = formattedMessage.replace('"', '')

        # add the message to the GUI
        self.messageLabel.text += formattedMessage


    def saveChatHistory(self, message):
        """
        Saves a message to the chat history json file
        Note: I might want to make it so you can save multiple messages at once, but for now it's just one at a time
        """

        chatHistory = {"logs": {}}
        try:
            with open(f"CharacterJsons/ChatHistory{sharedVars.player.name}.json", 'r') as f:
                chatHistory = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # If the chat history doesn't have a log for this character, create one
        chatHistory["logs"].setdefault(sharedVars.chatKey, []).append(message)
        
        # Update the chat history json file
        with open(f"CharacterJsons/ChatHistory{sharedVars.player.name}.json", 'w') as f:
            json.dump(chatHistory, f, indent=4)


    def sendMessage(self, *args):
        """
        Sends a message to the chatbot
        """
        userMessage = self.userInput.text.strip()
        if userMessage:
            message = {"role": "user", "content": userMessage}
            sharedVars.appendMessage(message)   # add message to global messages variable
            self.appendMessage(message)         # add message to gui    
            self.saveChatHistory(message)       # save message to chat history json file
            self.chatLoop()                     # get a response from the chatbot
            self.userInput.text = ""            # clear the user input box

    def chatLoop(self):
        """
        Gets a response from the chatbot
        """

        # make sure all the variables are up to date
        self.updateVars()
        messages = sharedVars.messages

        # If the last message is a new system message, display it in the gui
        lastMessage = messages[-1]
        if lastMessage["role"] == "system" and lastMessage["content"] not in self.messageLabel.text:
            self.appendMessage(lastMessage)

        # get the total number of tokens in the chat history
        totalTokens = sum(countTokens(message["content"]) for message in messages)
        print(f"Total tokens: {totalTokens}")
        
        # If the total number of tokens is greater than the token limit, remove messages until it's not
        while totalTokens > self.tokenLimit:
            print(f"Total tokens: {totalTokens}, Token limit: {self.tokenLimit}")
            
            # Remove the oldest message after the prompt and initial system message
            removedMessage = messages.pop(0)

            # If the prompt was removed, add it closer to the end of the list
            if "~!~!~" in removedMessage["content"] or removedMessage["role"] == "system":
                if len(messages) < 5:
                    messages.insert(-2, removedMessage)
                messages.insert(-5, removedMessage)
            else:
                # Update the total number of tokens
                totalTokens -= countTokens(removedMessage["content"])
        badResponse = True

        # Get a response from the chatbot and make sure it's not responding as the player
        while badResponse:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.maxTokens,
                top_p=self.topP,
                frequency_penalty=self.frequencyPenalty,
                presence_penalty=self.presencePenalty,
            )
            if f"{sharedVars.player.name}:" in response.choices[0].message.content:
                messages.append({"role": "system", "content": f"regenerate response. You're not allowed to respond as {sharedVars.player.name}"})
            else:
                badResponse = False

        # remove the messages from the bad responses
        for i in messages:
            if f"regenerate response. You're not allowed to respond as {sharedVars.player.name}" in i["content"]:
                messages.remove(i)

        responseMessage = response.choices[0].message   # separate the response from the rest of the response object
        self.saveChatHistory(responseMessage)           # save the response to the chat history json file
        sharedVars.appendMessage(responseMessage)       # add the response to the global messages variable
        self.appendMessage(responseMessage)             # add the response to the gui
