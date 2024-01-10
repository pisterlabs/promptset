# Importing the necessary Python libraries
import yaml
import openai
import gradio as gr



## OPENAI CONNECTION
## ---------------------------------------------------------------------------------------------------------------------
# Loading the API key and organization ID from file (NOT pushed to GitHub)
with open('../keys/openai-keys.yaml') as f:
    keys_yaml = yaml.safe_load(f)

# Applying our API key and organization ID to OpenAI
openai.organization = keys_yaml['ORG_ID']
openai.api_key = keys_yaml['API_KEY']

# Setting the OpenAI model selection (may adjust later to be user changeable for those lucky folks out there with GPT-4 access ;) )
openai_model = 'gpt-4'

# Setting the number of words to return in a response
NUM_WORDS = 300



## PROMPT ENGINEERING
## ---------------------------------------------------------------------------------------------------------------------
# Setting options to select from for philosophers
PHILOSOPHERS = [
    'Alan Watts',
    'Anne Lamott',
    'Brene Brown',
    'Duncan Trussell',
    'Eckhart Tolle',
    'Joseph Campbell',
    'Pete Holmes',
    'Ram Dass',
    'Scott Adams',
    'Socrates'
]

# Setting a list of comedians, just to be able to get better results from the model
COMEDIANS = [
    'Duncan Trussell',
    'Pete Holmes',
    'Scott Adams'
]



## GRADIO HELPER FUNCTIONS
## ---------------------------------------------------------------------------------------------------------------------
def converse_amongst_philosophers(philosopher_1, philosopher_2, convo_topic, convo_chatbot, rounds = 2):
    '''
    Simulates a conversation between two phiosopher using Generative AI

    Inputs:
        - philosopher_1 (str): The name of the first philosopher, who will begin the conversation
        - philosopher_2 (str): The name of the second philosopher
        - convo_topic (str): The topic of conversation about to take place between the philosophers
        - convo_chatbot (Gradio Chatbot): The chatbot interface that will hold the dialogue betweent the two philosophers
        - rounds (int): The number of rounds of conversation that will take place (default = 2)

    Returns:
        - convo_chatbot (Gradio Chatbot): The chatbot interface that holds the dialogue betweent the two philosophers
    '''

    # Checking if the philosopher is also a comedian
    if philosopher_1 in COMEDIANS:
        philosopher_1 = 'and comedian ' + philosopher_1
    if philosopher_2 in COMEDIANS:
        philosopher_2 = 'and comedian ' + philosopher_2

    # Instantiating chat flows for each respective philosopher
    philosopher_1_chat_flow = []
    philosopher_2_chat_flow = []

    # Prompt engineering the opening from philosopher 1
    philosopher_1_opener_prompt = f'''
    You are philosopher {philosopher_1} and are about to have a conversation with another philosopher, {philosopher_2}.
    The topic of conversation is {convo_topic}.
    You are first to speak.
    Please give your opening as {philosopher_1}
    Do not continue as {philosopher_2}.
    Please keep your opening under {NUM_WORDS} words.
    '''

    # Simulating the opening of the dialogue with philsopher 1 kicking things off
    philosopher_1_chat_flow.append({'role': 'user', 'content': philosopher_1_opener_prompt})
    philosopher_1_opener = openai.ChatCompletion.create(
        model = openai_model,
        messages = philosopher_1_chat_flow
    )['choices'][0]['message']['content']
    philosopher_1_chat_flow.append({'role': 'assistant', 'content': philosopher_1_opener})

    # Prompt engineering the opening from philosopher 2
    philosopher_2_opener_prompt = f'''
    You are philosopher {philosopher_2} and are about to have a conversation with another philosopher, {philosopher_1}.
    The topic of conversation is {convo_topic}.
    The other person has opened the conversation with the following:
    "{philosopher_1_opener}"
    Respond back accordingly.
    Do not continue as {philosopher_1}.
    Please keep your response under {NUM_WORDS} words.
    '''

    # Simulating the opening response from philosopher 2 on hearing philosopher 1's opening
    philosopher_2_chat_flow.append({'role': 'user', 'content': philosopher_2_opener_prompt})
    philosopher_2_response = openai.ChatCompletion.create(
        model = openai_model,
        messages = philosopher_2_chat_flow
    )['choices'][0]['message']['content']
    philosopher_2_chat_flow.append({'role': 'assistant', 'content': philosopher_2_response})

    # Appending the opening interaction to the chatbot
    convo_chatbot.append((philosopher_1_opener, philosopher_2_response))

    # Continuing a general back-and-forth based on number of rounds
    for _ in range(rounds):

        # Prompt engineering the continued conversation for philosopher 1
        philosopher_1_response_prompt = f'''
        {philosopher_2} has responded with the following:
        "{philosopher_2_response}"
        Respond back accordingly.
        Do not continue as {philosopher_2}.
        Plase keep your response under {NUM_WORDS} words.
        '''

        # Simulating the response from philosopher 1
        philosopher_1_chat_flow.append({'role': 'user', 'content': philosopher_1_response_prompt})
        philosopher_1_response = openai.ChatCompletion.create(
            model = openai_model,
            messages = philosopher_1_chat_flow
        )['choices'][0]['message']['content']
        philosopher_1_chat_flow.append({'role': 'assistant', 'content': philosopher_1_response})

        # Prompt engineering the continued conversation for philosopher 2
        philosopher_2_response_prompt = f'''
        {philosopher_1} has responded with the following:
        "{philosopher_1_response}"
        Respond back accordingly.
        Do not continue as {philosopher_1}.
        Plase keep your response under {NUM_WORDS} words.
        '''

        # Simulating the response from philosopher 2
        philosopher_2_chat_flow.append({'role': 'user', 'content': philosopher_2_response_prompt})
        philosopher_2_response = openai.ChatCompletion.create(
            model = openai_model,
            messages = philosopher_2_chat_flow
        )['choices'][0]['message']['content']
        philosopher_2_chat_flow.append({'role': 'assistant', 'content': philosopher_2_response})

        # Appending this round of conversation to the chatbot
        convo_chatbot.append((philosopher_1_response, philosopher_2_response))

    # Prompt engineering a close of the conversation instigated by philosopher 1
    philosopher_1_closer_prompt = f'''
    {philosopher_2} has responded with the following:
    "{philosopher_2_response}"
    It's time to bring this conversation to a close. Please give one final thought before closing.
    Do not continue as {philosopher_2}.
    Please keep your response under {NUM_WORDS} words.
    '''

    # Simulating the closer from philosopher 1
    philosopher_1_chat_flow.append({'role': 'user', 'content': philosopher_1_closer_prompt})
    philosopher_1_closer = openai.ChatCompletion.create(
        model = openai_model,
        messages = philosopher_1_chat_flow
    )['choices'][0]['message']['content']
    philosopher_1_chat_flow.append({'role': 'assistant', 'content': philosopher_1_closer})

    # Prompt engineering a close of the conversation, finally wrapping things up with philosopher 2
    philosopher_2_closer_prompt = f'''
    {philosopher_1} is bringing the conversation to a close with this final remark:
    "{philosopher_1_closer}"
    Please bring this conversation to a close and keep your response under {NUM_WORDS} words.
    '''

    # Simulating the closer from philosopher 2
    philosopher_2_chat_flow.append({'role': 'user', 'content': philosopher_2_closer_prompt})
    philosopher_2_closer = openai.ChatCompletion.create(
        model = openai_model,
        messages = philosopher_2_chat_flow
    )['choices'][0]['message']['content']
    philosopher_2_chat_flow.append({'role': 'assistant', 'content': philosopher_2_closer})

    # Appending the closing remarks to the chatbot
    convo_chatbot.append((philosopher_1_closer, philosopher_2_closer))

    return convo_chatbot



## GRADIO UI LAYOUT & FUNCTIONALITY
## ---------------------------------------------------------------------------------------------------------------------
# Defining the building blocks that represent the form and function of the Gradio UI
with gr.Blocks(title = 'Philosophy Conversation Simulator', theme = 'base') as convo_sim:
    
    # Setting the overall header for the page
    gr.Markdown('''
    # Philosophy Conversation Simulator
    
    This interface allows you to simulate a conversation between two philosophers about whatever you want them to talk about!
    ''')

    # Setting a side-by-side selector for conversators
    with gr.Row():

        with gr.Column():

            # Enabling a dropdown to select the first participant
            philosopher_1 = gr.Dropdown(choices = PHILOSOPHERS, label = 'Philosopher 1', allow_custom_value = True)

        with gr.Column():
            # Enabling a dropdown to select the second participant
            philosopher_2 = gr.Dropdown(choices = PHILOSOPHERS, label = 'Philosopher 2')

    # Creating a freeform textbox allowing the user to submit any topic they would like the participants to converse about
    convo_topic = gr.Textbox(label = 'Please enter an idea for a topic of conversation.',
                             placeholder = 'e.g. Chicago Style Pizza')
    
    # Creating the button to simulate the conversation
    simulate_conversation_button = gr.Button('Simulate Conversation')

    # Instantiating the chatbot interface to hold the back-and-forth of the conversation
    convo_chatbot = gr.Chatbot(label = 'Simulated Conversation')

    # Defining the behavior for when the user clicks the "Simulate Conversation" button
    simulate_conversation_button.click(fn = converse_amongst_philosophers,
                                       inputs = [philosopher_1, philosopher_2, convo_topic, convo_chatbot],
                                       outputs = [convo_chatbot],
                                       queue = False)




## SCRIPT INVOCATION
## ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Instantiating the initial chat flow used as a global variable
    chat_flow = []

    # Launching the Gradio UI
    convo_sim.launch()