import json
from langchain.memory import ChatMessageHistory
from colorama import Fore as c

def new_session(session_id:int=None, name:str=None): 
    '''
    Method to create a new session. Where:
        - session_id: id of the session
        - name: name of the session
    '''

    # Read sessions.json
    with open(f"./data/sessions.json", "r") as f:
        data = json.load(f)
    
    # Set ActiveSession to session_id
    data["Active"] = session_id
    # Add session_id to sessions
    data[str(session_id)] = {
        "name":name,
        "history":[]
    }

    return data

def load_session(session_id:int=None):
    '''
    Method to load a existing session. Where:
        - session_id: id of the session
    '''
    
    # Load sessions.json
    with open(f"./data/sessions.json", "r") as f:
        data = json.load(f)

    # Check if session_id is exists in sessions.json
    if str(session_id) in data.keys():
        print(f'\n[ {c.GREEN}SESSION{c.RESET} ] Session exists')
        history = data[str(session_id)]["history"]

        # Create a new ChatMessageHistory object
        session = ChatMessageHistory()
        # Add history to session
        speaker = 0
        for i in history:
            if speaker == 0:
                new_i = i[len("Human:"):].strip()
                session.add_user_message(new_i)
                speaker = 1
            else:
                new_i = i[len("AI:"):].strip()
                session.add_ai_message(new_i)
                speaker = 0

        return session
    else:
        print(f'\n[ {c.YELLOW}SESSION{c.RESET} ] Session does not exist')
        session = ChatMessageHistory()
        return session

def check_session():
    '''
    Method to check the active session in sessions.json
    '''

    with open(f"./data/sessions.json", "r") as f:
        data = json.load(f)

    active = data["Active"]

    return active

def set_session(session_id:int=None):
    '''
    Method to set the active session in sessions.json. Where:
        - session_id: id of the session
    '''
    # Change active session in sessions.json
    with open(f"./data/sessions.json", "r") as f:
        data = json.load(f)
        data["Active"] = session_id

    # Save sessions.json
    with open(f"./data/sessions.json", "w") as f:
        json.dump(data, f, indent=4)

def save_session(history:list=None):
    '''
    Method to save the history of the session in sessions.json. Where:
        - history: history of the actual session
    '''

    # Check the last session id in sessions.json
    try:
        with open(f"./data/sessions.json", "r") as f:
            data = json.load(f)
            
        session_id = int(data["Active"])

        new_history = []

        speaker = 0
        for part in history:
            if speaker == 0:
                new_history.append('Human: ' + part.content)
                speaker = 1
            else:
                new_history.append('AI: ' + part.content)
                speaker = 0
        
        name = new_history[0]
        name = name[len("Human:"):].strip()
        if len(name) > 30:
            name = name[:30] + '...'

        if str(session_id) not in data.keys():
            print(f'\n[ {c.CYAN}SESSION{c.RESET} ] Session does not exist. Creating new session...')
            data = new_session(session_id=session_id, name=name)
            data[str(session_id)]["history"] = new_history
        else:
            print(f'\n[ {c.CYAN}SESSION{c.RESET} ] Session exists. Saving history...')
            data[str(session_id)]["history"] = new_history

        # Save sessions.json
        with open(f"./data/sessions.json", "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    except Exception as e:
        print(f"[ {c.RED}SESSION{c.RESET} ] Error saving session: {e}")
        return 'Error saving session.'