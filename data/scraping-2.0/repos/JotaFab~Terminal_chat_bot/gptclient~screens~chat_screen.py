import os
from textual import events
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Input, RichLog, Header,Button,Footer
from textual.binding import Binding
from time import sleep
from openai import OpenAI
from textual import log,on,events
from gptclient.screens.assistants_screen import AssistantsScreen
from gptclient.database import *
from gptclient.methods import * 
from textual.reactive import reactive


import asyncio

def valid_api_key(api_key: str) -> bool:
    """Check if the api_key is valid"""
    client = OpenAI(api_key=api_key)
    try:
        client.models.list()
        return True
    except:
        return False
    
    
    
def get_api_key() -> str:
    """Get the OpenAI API key from the environment variable OPENAI_API_KEY if it exists, otherwise prompt the user for it."""
    api_key = None
    
    if os.environ.get("OPENAI_API_KEY"):
        api_key = os.environ["OPENAI_API_KEY"]
        if valid_api_key(api_key):
            return api_key
        else:
            api_key = None

    return api_key  

                

class ChatScreen(Screen):
    
    CSS_PATH=Path(__file__).parent / "style/chat_style.tcss"
    
    BINDINGS = [
        Binding("ctrl+n", action="new_chat()", description="New Chat"),
        Binding(key="ctrl+shift+enter", action="send_message", description="Send Message")
    ]
    
    def __init__(self):
        super().__init__()
        
        
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield RichLog(highlight=True,markup=True, name="chat-box")
        yield Input(id="chat-input")
        yield Button("Send", id="send-message")
        yield Footer()   
        
    def on_mount(self):
        self.api_key = get_api_key()
        log(self.api_key)
        self.client = OpenAI(
            api_key=self.api_key
        )
        log(self.client)
        self.text_log = self.query_one(RichLog)
        self.assistant=retrieve_assistant(self.client,self.app.assistant_id)
        self.thread = create_thread(self.client)
        self.thread_id=self.thread.id
        self.usr_name=self.app.usr_name
        self.message = None
        log(self.assistant)
        
    
    def action_new_chat(self) -> None:
        self.thread = create_thread(self.client, self.assistant_id)
        self.text_log.clear()
        self.thread=create_thread(self.client)
        self.thread_id=self.thread.id
            
    @on(Input.Changed, "#chat-input")
    def chat_input_changed(self, event: Input.Changed) -> None:
        self.message = str(event.value)
          
    @on(Button.Pressed, "#send-message")            
    def action_send_message(self) -> None:
        """Send a message to the assistant and return the response"""
        self.text_log.write(f"{self.usr_name}: \n")
        self.text_log.write(self.message)
        self.text_log.write(f"\n{self.assistant.name}: \n")
        promtp = self.message
        Input.clear(self)
        response = self.send_message(promtp)
        log(response)
        response = response.value
        self.text_log.write(f"{response}")
        
    
    
    def send_message(self,promtp):
        """Send a message to the assistant and return the response"""
        create_message(client=self.client, thread_id=self.thread.id, message=promtp)
        run = create_run(client=self.client, thread_id=self.thread.id, assistant_id=self.assistant.id)
        sleep(1)
        run_retrieve = retrieve_run(client=self.client, thread_id=self.thread.id, run_id=run.id)
        steps_list = list_run_steps(client=self.client,thread_id=self.thread.id,run_id=run.id)
        while not steps_list.data or steps_list.data[0].status != "completed":
            sleep(1)
            steps_list = list_run_steps(client=self.client,thread_id=self.thread.id,run_id=run.id)
            
        log(steps_list.data[0].status)
        
        response = retrieve_message(client=self.client,thread_id=self.thread.id,message_id=steps_list.data[0].step_details.message_creation.message_id)
        return response

        