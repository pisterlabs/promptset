import subprocess
import os
from sys import platform
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


def latex_compile(path):
    try:
        subprocess.run(['pdflatex', path])
    except subprocess.CalledProcessError as err:
        print(f"LaTeX compilation failed with error: {err}")

#sometimes it make the file perfect but adds a random word infront of everything causing it to be uncompileable, this checks the first line and fixes if it neccesary
def first_line_fix(filename, starting_line):
    with open(filename, 'r+') as file:
        first_line = file.readline()
        if not first_line.startswith(starting_line):
            lines = file.readlines()
            file.seek(0)
            file.writelines(lines[1:])
            file.truncate()

starting_line = r'\documentclass{article}'


role = r"""
  You are LaTeCh. A large language model trained to provided structred .tex files ready for latex compilation. 

  You notes should always be extensive and highly detailed. Your notes must be at least 1000 words long, or more if the topic is complex, like quantum physics, cryptography, artifcial intelligence and such. You must includ examples and equations.

! IMPORTANT !
    FOR MATH AND PHYSICS QUESTIONS PROVIDE EQUATIONS
    FOR PROGRAMING QUESTIONS PROVIDE CODE SAMPLES
! IMPORTANT !

All your writings must be in a .tex format, one that is able to be compiled with pdflatex.

! IMPORTANT !
Always begin the OUTPUT with :
\documentclass{article}
\begin{document}
! IMPORTANT !

Be sure to add the appropriate '\usepackage' that are required for said notes.

Example input : "Newton's first law"

Example output : 
"
\documentclass{article}
\usepackage{amsmath}
\begin{document}

\section*{Newton's First Law}

Newton's first law, also known as the law of inertia, states that an object at rest will remain at rest, and an object in motion will continue in motion with a constant velocity unless acted upon by an external force.

\subsection*{Equations}

Here are some equations related to Newton's first law:

\begin{enumerate}
  \item The equation of motion for a particle under no external force:
  \begin{equation*}
    \sum F = 0
  \end{equation*}
  
  \item The equilibrium condition for forces acting on an object:
  \begin{equation*}
    \sum F_{\text{net}} = 0
  \end{equation*}
  
  \item The relationship between force, mass, and acceleration:
  \begin{equation*}
    F = m \cdot a
  \end{equation*}
\end{enumerate}

\end{document} 
"


"""


prompt = input ("What notes would you like ? ")


tex_file_path = prompt.strip()+".tex"

model_prompt=[{"role": "user", "content": role+prompt}]


notes = openai.ChatCompletion.create (
    model="gpt-3.5-turbo-16k",
    messages = model_prompt,
    temperature=0.5,
    max_tokens=7500,
    frequency_penalty=0.0
)



with open (tex_file_path,"w") as file :
        file.write(notes.choices[0].message.content.strip())

first_line_fix(tex_file_path, starting_line)

latex_compile(tex_file_path)
latex_file = prompt + ".pdf"


if platform.startswith('win'):
        subprocess.run([latex_file])
else :
        subprocess.run(['open',latex_file])
