import openai
import subprocess
import gradio as gr
import re
import spacy 
import ast

# load and set our key
openai.api_key = open("key.txt", "r").read().strip("\n")

completion = openai.ChatCompletion.create(
  model="gpt-4", # this is "ChatGPT" $0.002 per 1k tokens
  messages=[{"role": "user", "content": "How do we make our model on our own question answering dataset for RFP(Request for proposal) operations based on past responses to automate question answering in companies using techniques that you would suggest. Detail your answers. Atleast Minimum 1000 words "}]
)

content = completion['choices'][0]['message']['content']
content


code_block_regex = r'```python\n([\s\S]+?)\n```'

# Extract the code block(s) from the text using the regular expression
code_blocks = re.findall(code_block_regex, content)

# Parse the code block(s) using the ast module
print(code_blocks)
for code in code_blocks:
    try:
        parsed_code = ast.parse(code.strip())
        print(ast.dump(parsed_code))
    except SyntaxError as e:
        print(f"Error parsing code: {e}")
        
import spacy

# Load the English language model for spaCy
nlp = spacy.load('en_core_web_sm')

# Define the text to be parsed
text = "Once upon a time, in a forest far away, there lived a group of animals. They were all different shapes and sizes, but they all shared one thing in common - they loved to learn.\n\nOne day, as they were exploring the forest, they came across a group of children who were out on a nature walk. The animals were thrilled to see the children and wanted to show them all the amazing things they had learned about the forest.\n\nFirst, they introduced the children to the trees. The tall, strong trees provided shelter and food for many animals, and some of them even produced delicious fruits and nuts that the animals loved to eat."

# Parse the text using spaCy
doc = nlp(text)

# Loop through each sentence in the parsed text
for sentence in doc.sents:
    # Print the sentence text with commas replaced by line breaks
    print(sentence.text.replace(',', '\n'))

    
# TO DO : Add text to speech model here 



# Assuming you want to run the 'mycode.py' file saved earlier
completed = subprocess.run(['python', 'mycode.py'],capture_output=True, text=True)
completed.stdout

import ast

content = completion['choices'][0]['message']['content']
lines = content.split('\n')

# Extract the Python code blocks from the lines
code_blocks = []
for i in range(len(lines)):
    line = lines[i].strip()
    if line.startswith('```python') or line.startswith('import'):
#         print("IN here???")
        code_block = ''
        i += 1
        while i < len(lines) and not lines[i].startswith('```'):
            code_block += lines[i] + '\n'
            i += 1
        code_blocks.append(code_block)

print(code_blocks)
# # Run the Python code blocks
# for code_block in code_blocks:
#     code_block = ast.literal_eval(code_block.strip('```').strip())
#     exec(code_block)
    
  


openai.api_key = open("key.txt", "r").read().strip("\n")

message_history = [{"role": "user", "content": f"You are a joke bot. I will specify the subject matter in my messages, and you will reply with a joke that includes the subjects I mention in my messages. Reply only with jokes to further input. If you understand, say OK."},
                   {"role": "assistant", "content": f"OK"}]

def predict(input):
    # tokenize the new input sentence
    message_history.append({"role": "user", "content": f"{input}"})

    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", #10x cheaper than davinci, and better. $0.002 per 1k tokens
      messages=message_history
    )
    #Just the reply:
    reply_content = completion.choices[0].message.content#.replace('```python', '<pre>').replace('```', '</pre>')

    print(reply_content)
    message_history.append({"role": "assistant", "content": f"{reply_content}"}) 
    
    # get pairs of msg["content"] from message history, skipping the pre-prompt:              here.
    response = [(message_history[i]["content"], message_history[i+1]["content"]) for i in range(2, len(message_history)-1, 2)]  # convert to tuples of list
    return response

# creates a new Blocks app and assigns it to the variable demo.
with gr.Blocks() as demo: 

    # creates a new Chatbot instance and assigns it to the variable chatbot.
    chatbot = gr.Chatbot() 

    # creates a new Row component, which is a container for other components.
    with gr.Row(): 
        '''creates a new Textbox component, which is used to collect user input. 
        The show_label parameter is set to False to hide the label, 
        and the placeholder parameter is set'''
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
    '''
    sets the submit action of the Textbox to the predict function, 
    which takes the input from the Textbox, the chatbot instance, 
    and the state instance as arguments. 
    This function processes the input and generates a response from the chatbot, 
    which is displayed in the output area.'''
    txt.submit(predict, txt, chatbot) # submit(function, input, output)
    #txt.submit(lambda :"", None, txt)  #Sets submit action to lambda function that returns empty string 

    '''
    sets the submit action of the Textbox to a JavaScript function that returns an empty string. 
    This line is equivalent to the commented out line above, but uses a different implementation. 
    The _js parameter is used to pass a JavaScript function to the submit method.'''
    txt.submit(None, None, txt, _js="() => {''}") # No function, no input to that function, submit action to textbox is a js function that returns empty string, so it clears immediately.
         
demo.launch()
