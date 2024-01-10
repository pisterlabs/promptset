
import os
import threading
from Buddy import FinnAGI
from langchain.chat_models import ChatOpenAI
from workspace.tools.base import AgentTool
from workspace.toolbag.toolbag import Toolbag
from ui.cui import CommandlineUserInterface
import PySimpleGUI as sg
import importlib
from dotenv import load_dotenv
load_dotenv()
main_stop_event = threading.Event()  # Event object to stop the main script
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"
AGENT_DIRECTORY = os.getenv("AGENT_DIRECTORY", "")
assert AGENT_DIRECTORY, "AGENT_DIRECTORY variable is missing from .env"
### 1.Create Agent ###
dir = AGENT_DIRECTORY
script_dir = os.path.dirname(os.path.abspath(__file__))
window = None
def main():
    try:
        finnagi = FinnAGI(
    ui=CommandlineUserInterface(),
    openai_api_key=OPENAI_API_KEY,
    dir=dir,
)
        tools = []
        toolbag = Toolbag()
        for tool in toolbag.toolbag:
            tools.append(tool)
        for tool in tools:
            module_name = tool['class']  # module name
            function_name = tool['func'].split('.')[-1]  # function name

            # load the module
            module = importlib.import_module(f"workspace.tools.{module_name}")

            # get the function
            classe = getattr(module, module_name)
            instance = classe()
            func = getattr(instance, function_name)

            #print(tool)
            tool_instance = AgentTool(
                name=tool['name'],
                func=func,
                args=tool['args'],
                description=tool['description'],
                user_permission_required=tool['user_permission_required']
            )
            print(tool_instance)
            finnagi.procedural_memory.memorize_tools([tool_instance])
        while not main_stop_event.is_set():
            try:
                window.write_event_value('-UPDATE-', (str(finnagi.thoughts), str(finnagi.working_memory)))
                finnagi.run()
            except KeyboardInterrupt:
                print("Exiting...")
            pass

            # Continue doing what the AGI was doing...
    except KeyboardInterrupt:
        print("Exiting...")


def start_main():
    threading.Thread(target=main, daemon=True).start()


def selectSettings():
    global window
    layout = [
        [
            [sg.Text('Game Window', size=(15, 1))],
            [sg.Button('Start'), sg.Button('Stop'), sg.Button('Exit')],
            [sg.Text('Thoughts:', size=(10,1)), sg.Text('', key='-THOUGHTS-', size=(100,5))], # Display FinnAGI.thoughts
            [sg.Text('Working Memory:', size=(15,1)), sg.Text('', key='-WORKING_MEMORY-', size=(100,5))], # Display FinnAGI.workingmemory
        ]
    ]

    
    [sg.Text("Chat with FinnAGI", font=("Helvetica", 14))],
    [sg.Multiline(size=(50, 10), key="-CHATBOX-", disabled=True, autoscroll=True)],
    [sg.InputText(size=(40, 1), key="-USERINPUT-"), sg.Button("Send", bind_return_key=True)],
window = sg.Window("Proton Client", layout)
    main_thread = None

    while True:
        event, values = window.read()

        if event == 'Start':
            if main_thread is None or not main_thread.is_alive():  # Check if the thread is not running
                main_stop_event.clear()  # Clear the stop event
                main_thread = threading.Thread(target=main, daemon=True)
                main_thread.start()
        elif event == 'Stop':
            print('Stopping script.')
            main_stop_event.set()  # Set the stop event
        elif event == '-UPDATE-':  # Event to update FinnAGI values
            thoughts, working_memory = values[event]  # Unpack the values
            window['-THOUGHTS-'].update(thoughts)
            window['-WORKING_MEMORY-'].update(working_memory)
        elif event == 'Exit' or event == sg.WIN_CLOSED:
            main_stop_event.set()  # Set the stop event
                if event == "Send":
            # Capture user's message from the input box
            user_message = values["-USERINPUT-"]
            
            # Append user's message to chatbox
            window["-CHATBOX-"].print(f"You: {user_message}")
            
            # Get a response from FinnAGI (You should replace this with actual logic from Buddy.py)
            agent_response = "Agent: " + user_message[::-1]  # Reverse the message as a dummy response
            
            # Append agent's message to chatbox
            window["-CHATBOX-"].print(agent_response)
            
            # Clear user input
            window["-USERINPUT-"].update("")
    break
    window.close()

if __name__ == "__main__":
    selectSettings()





