import sys

sys.path.insert(0, '/Users/shreyagarwal/Code/GitHub/MATH-GPT/declarative-math-word-problem')
sys.path.insert(0, '/Users/shreyagarwal/Code/GitHub/MATH-GPT/declarative-math-word-problem/prompts')
sys.path.insert(0, '/Users/shreyagarwal/Code/GitHub/MATH-GPT/pal')
sys.path.insert(0, '/Users/christinaxu/Documents/GitHub/declarative-math-word-problem')
sys.path.insert(0, '/Users/christinaxu/Documents/GitHub/declarative-math-word-problem/prompts')
sys.path.insert(0, '/Users/christinaxu/Documents/GitHub/math-gpt/pal')
import tkinter as tk
from tkinter.ttk import *
from utils import *
from declarative_three_shot import DECLARATIVE_THREE_SHOT_AND_PRINCIPLES
import openai
import pal
from pal.prompt import math_prompts
from langchain.chains import PALChain
from langchain import OpenAI, LLMMathChain
from langchain.chains import PALChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
import os

# access the api key from whatever file you have it in
def get_file_contents(filename):
    try:
        with open(filename, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found" % filename)

api_key = get_file_contents('api_key.txt')
openai.api_key = api_key

root= tk.Tk()
root.title('math-gpt')
root.resizable(False, False)

# initializing the pal model and interface
MODEL = 'text-davinci-003' #@param {type:"string"}m
interface = pal.interface.ProgramInterface(model=MODEL, get_answer_expr='solution()', verbose=True)

# initializing the chat-gpt prompter for vanilla da-vinci
def chat_with_gpt(prompt):
    response=openai.Completion.create(
        engine='text-davinci-003',  # Choose the appropriate GPT model
        prompt=prompt,
        max_tokens=100,  # Adjust as needed to control the response length
        temperature=0.7,  # Adjust as needed to control the response randomness
    )
    return response.choices

# initializing all of the langchain models and tools 
os.environ["OPENAI_API_KEY"] = api_key
llm = OpenAI(temperature=0, model_name="text-davinci-003")
tools = load_tools(["llm-math"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,
)

# initialize the canvas
canvas1 = tk.Canvas(root, width=750, height=750, bg = "white")
canvas1.pack()

# title text
title_text = tk.Label(canvas1, bg="white", fg="black", height=1, width=10, font=("Times New Roman", 36))
title_text.place(relx=0.5, y=30, anchor="center")
title_text.config(text='Math-GPT')

names_text = tk.Label(canvas1, bg="white", fg="black", height=1, width=53, font=("Times New Roman", 20))
names_text.place(relx=0.5, y=60, anchor="center")
names_text.config(text='Shrey Agarwal, Christina Xu, Hamid Bagheri, Lisong Xu')

prompt_label = tk.Label(canvas1, bg="white", fg="black", height=1, width=50, font=("Times New Roman", 20))
prompt_label.place(relx=0.5, y=100, anchor="center")
prompt_label.config(text='Enter Prompt:')

method_label = tk.Label(canvas1, bg="white", fg="black", height=1, width=50, font=("Times New Roman", 20))
method_label.place(relx=0.5, y=200, anchor="center")
method_label.config(text="Choose your method after you've entered your prompt:")

answer_label = tk.Label(canvas1, bg="white", fg="black", height=1, width=50, font=("Times New Roman", 20))
answer_label.place(relx=0.5, y=275, anchor="center")
answer_label.config(text="The answer will be displayed here:")

explanation_label = tk.Label(canvas1, bg="white", fg="black", height=1, width=100, font=("Times New Roman", 14))
explanation_label.place(relx=0.5, y=340, anchor="center")
explanation_label.config(text="For the Vanilla DaVinci, LangChain, and the Symbolic Solver, an explanation will be provided here:")

# create the entry box
entry1 = tk.Text(root, height = 3, font = ('Times New Roman', 16), bg = "white", fg = "black")
#entry1.pack(padx = 10, pady = 10)
entry1.place(relx=0.5, y = 150, anchor="center")

#entry1 = tk.Entry(width=50, font=("Arial 16"), bg="white", fg="black", justify='center')
#entry1.pack(padx=10, pady=10)
#entry1.place(relx=0.5, y = 150, anchor="center")

# function to call for using vanilla davinci
def use_vanilla_davinci():  
    expression = entry1.get("1.0", 'end-1c')
    # finding the answer
    prompt_for_answer = f"Given the expression '{expression}', please generate an answer."
    response_choices = chat_with_gpt(prompt_for_answer)
    answer = [choice.text.strip() for choice in response_choices]
    # finding the explanation
    prompt_for_explanation = f"Given the expression '{expression}', please write a solution that correctly addresses the problem and solves it."
    response_choices_for_explanation = chat_with_gpt(prompt_for_explanation)
    answer_for_explanation = [choice.text.strip() for choice in response_choices_for_explanation]
    # configuring labels to display answer and explanation
    explanation_text.config(text="")
    answer_text.config(text=f"Vanilla answer: '{answer}'.")
    explanation_text.config(text=answer_for_explanation)

# function to call for using langchain
def use_langchain():  
    x1 = entry1.get("1.0", 'end-1c')
    llm = OpenAI(temperature = 0)
    llm_math = LLMMathChain.from_llm(llm, verbose = True)
    answer = llm_math.run(x1)
    response = agent(
        {
            "input": x1
        }
    )
    l = response["intermediate_steps"]
    # manipulating AgentAction namedTuple to find answer and explanation
    if len(l) >= 2:
        answer = ""
        explanation = ""
        answer = (str(l[len(l) - 1][1]))
        for i in l:
            explanation += str(i[0]).split(", ", 2)[2][6:-2]
    else:
        list = l[0]
        answer = ""
        explanation = ""
        answer = str(list[1])
        explanation = str(list[0]).split(", ", 2)[2][6:-2]
    explanation_text.config(text="")
    answer_text.config(text=answer)
    explanation_text.config(text=explanation)

# function to call for using PAL
def use_pal():  
    x1 = entry1.get("1.0", 'end-1c')
    prompt = math_prompts.MATH_PROMPT.format(question=x1)
    answer = interface.run(prompt)
    explanation = ""
    explanation_text.config(text="")
    answer_text.config(text=answer)
    explanation_text.config(text=explanation)

# function to call for using the symbolic solver
def use_symbolic_solver():  
    x1 = entry1.get("1.0", 'end-1c')
    eq_list = get_declarative_equations(model='text-davinci-003', question=x1, prompt=DECLARATIVE_THREE_SHOT_AND_PRINCIPLES, max_tokens=600, stop_token='\n\n\n', temperature=0)
    answer = get_final_using_sympy(eq_list)
    explanation_text.config(text="")
    answer_text.config(text=f"Symbolic Solver answer: '{answer}'.")
    explanation_text.config(text=eq_list)

# creating all the buttons and the answer text
button1 = tk.Button(bg = "white", font=("Times New Roman", 18), text='Vanilla DaVinci', borderwidth = 0, relief = "groove", command=use_vanilla_davinci)
canvas1.create_window(212, 240, window=button1)

button2 = tk.Button(bg = "white", font=("Times New Roman", 18), text='LangChain', borderwidth = 0, relief = "groove", command=use_langchain)
canvas1.create_window(345, 240, window=button2)

button3 = tk.Button(bg = "white", font=("Times New Roman", 18), text='PAL', borderwidth = 0, relief = "groove", command=use_pal)
canvas1.create_window(435, 240, window=button3)

button3 = tk.Button(bg = "white", font=("Times New Roman", 18), text='Symbolic Solver', borderwidth = 0, relief = "groove", command=use_symbolic_solver)
canvas1.create_window(546, 240, window=button3)

answer_text = tk.Label(canvas1, bg="white", fg="black", height=1, width=65, font=("Times New Roman", 18), borderwidth = 3, relief = "groove")
answer_text.place(relx=0.5, y=310, anchor="center")

explanation_text = tk.Label(canvas1, bg="white", fg="black", height=10, width=65, font=("Times New Roman", 18), wraplength=300, justify='center', borderwidth = 3, relief = "groove")
explanation_text.place(relx=0.5, y=460, anchor="center")

root.mainloop()