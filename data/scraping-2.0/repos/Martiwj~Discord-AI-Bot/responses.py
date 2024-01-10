from openAi import get_chat_completion

def handle_response(message: str) -> str:
    p_message: str = message.lower()
    
    if p_message == 'hello':
        return 'Hi, im your personal cookbot'

    if p_message == 'help':
        return 'How can i help you today?'
    
    if p_message[0] == '?' :

        return get_chat_completion(p_message[1:])
    
     if p_message[0] == '#':
        return create_image(p_message[1:])
