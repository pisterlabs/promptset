from langchain.prompts import PromptTemplate

## PROMPT CELL

QUERY_PROMPT = PromptTemplate(
                        input_variables=["question", "units"],
                        template="""{question} (units: {units})"""
                    )

CONTEXT_PROMPT = PromptTemplate(
                        input_variables=["context"],
                        template="""CONTEXT:=\n{context}"""
                    )

ASSISTANT_FIRST_RESP_PROMPT = PromptTemplate(
                        input_variables=["summary", "program"],
                        template="""\nSUMMARY:= {summary}\nPROGRAM:=\n{program}\n"""
                    )

ASSISTANT_SECOND_RESP_PROMPT = PromptTemplate(
                        input_variables=["revised_program"],
                        template="""\nREVISED_PROGRAM:=\n{revised_program}\n"""
                    )


SYSTEM_PROMPT_STRING = """
You are a helpful assistant tasked with answering estimation questions.
The user will first ask an estimation question to answer and at the end of the question the user will indicate the units that they want the answer to be in or dimensionless if the answer has no units.

Your initial answer will contain two parts: a summary and a program.
SUMMARY:= First you should write a concise summary of any hypothetical scenarios which the question sets up as well as what is the exact quantity that the question is asking you to estimate.
PROGRAM:=
Next you must write a Python program that answers the estimation question and stores that answer in the variable `A0`.
Your program will have the following structure:
# Q_: Your program should ask sub-questions to help you answer the question.
# R_: After asking each sub-question, you must provide your reasoning for why the sub-question is relevant to our goal of answering the main question.
... You are then encouraged to decompose the sub-question even further unless you are confident that you know the numerical answer to the sub-question ...
# F_: Before answer each sub-question, you will provide a fact that you are referencing to answer the sub-question. The fact must be verifiable by a third party.
A_= You will then answer these sub-questions by assigning the answer to the variable `A_` where the number after the `A` corresponds to the number after the `Q` of the sub-question. # then put the units in a comment after the variable assignment
Please decompose all sub-questions as much as possible to ensure that all quantities are estimated correctly and the math to recompose the quantities is as simple as possible.
Ensure that all quantities use units which are consistent with the units for the final answer.
Additionally, make sure all numerical quantities used in your Python code are written in scientific notation.

After you provide this initial answer, the user will then provide a list of contextual information which they want you to integrate into your answer.
CONTEXT:=
The user will provide this contextual information as a bullet point list of facts.

Your response to the user will be a revision of your initial program.
REVISED PROGRAM:=
This program will have the exact same structure as your initial program with questions (Q_), reasoning (R_), facts (F_), and answers (A_) but you will revise the program to integrate the new contextual information provided by the user.
For every question, reasoning and fact used in your revised program, you must cite whether it is from the original program (Same as original) or from the new contextual information provided by the user (Revised based on new context).
If you are reusing a question, reasoning or fact from the original program, it does not have to be the same location that it appears the original program but you must cite where it is located in the original program.
Note that not all of the contextual information provided by the user will be relevant so you do not need to integrate all of it into your revised program.
You are encouraged to keep as much of your original program as much as possible but you are allowed to introduce new sub-questions and answer them if they are necessary to answer the main question given the new contextual information.
Additionally, you do not need to keep all sub-questions from your original program if they become unnecessary to the estimation question given the new contextual information.

In the following messages, you are provided with verified examplar conversations between the user and the assissant.
Lastly, you will be provided with a user question and you must continue the conversation in the same format as the previous examplar conversations.
"""

FEW_SHOT_PROMPTS = [
"""User:If I were to take an English penny, melt it down, then somehow stretch it out into a perfect hollow sphere 1 atom thick, how large would this sphere be? (units: in**3)

Assistant:
SUMMARY:= We take a penny, melt it down, and stretch it out into a hollow sphere 1 atom thick. We want to compute the volume of this sphere.
PROGRAM:=
# Q0: If I were to take an English penny, melt it down, then somehow stretch it out into a perfect hollow sphere 1 atom thick, how large would this sphere be?
# R0: This is the main question we want to answer.
# Q1: What is the volume of a single penny?
# R1: To compute the volume of the hollow sphere, we need the original volume of the penny since it is what gets melted down and turned into the sphere.
# F1: The volume of a penny is 0.84 in**3.
A1=8.4e-1 # (in**3)
# Q2: What is the surface area of a hollow sphere 1 atom thick which is constructed from a melted penny?
# R2: We only know the volume of the penny and we need to translate that information into information about the sphere.
# F2: The surface area of a hollow sphere 1 atom thick constructed from a melted penny is the same as the volume of the penny.
A2=8.4e-1 # (in**2)
# Q3: What is the radius of the sphere given its surface area?
# R3: To compute the volume of the sphere given its surface area, we first need to compute the radius of the sphere.
# F3: The radius of a sphere is the square root of the surface area of the sphere divided by 4*pi.
A3=(A2/(4*np.pi))**(1/2) # (in)
# F0: The volume of a sphere is 4/3*pi*r**3 where r is the radius of the sphere.
A0=(4/3)*np.pi*(A3 ** 3) # (in**3)

User:CONTEXT:=
- The volume of a penny is 12 cm**3.

Assistant:
REVISED PROGRAM:=
# Q0: If I were to take an English penny, melt it down, then somehow stretch it out into a perfect hollow sphere 1 atom thick, how large would this sphere be? (Same as original question Q0)
# R0: This is the main question we want to answer. (Same as original reasoning R0)
# Q1: What is the volume of a single penny? (Same as original question Q1)
# R1: To compute the volume of the hollow sphere, we need the original volume of the penny since it is what gets melted down and turned into the sphere. (Same as original reasoning R1)
# F1: The volume of a penny is 12 cm**3. We need to convert this into in**3 by multiplying by 0.0610237. (Revised fact based on new context)
A1=0.0610237 * 1.2e+1 # (in**3)
# Q2: What is the surface area of a hollow sphere 1 atom thick which is constructed from a melted penny? (Same as original question Q2)
# R2: We only know the volume of the penny and we need to translate that information into information about the sphere. (Same as original reasoning R2)
# F2: The surface area of a hollow sphere 1 atom thick constructed from a melted penny is the same as the volume of the penny. (Same as original fact F2)
A2=A1 # (in**2)
# Q3: What is the radius of the sphere given its surface area? (Same as original question Q3)
# R3: To compute the volume of the sphere given its surface area, we first need to compute the radius of the sphere. (Same as original reasoning R3)
# F3: The radius of a sphere is the square root of the surface area of the sphere divided by 4*pi. (Same as original fact F3)
A3=(A2/(4*np.pi))**(1/2) # (in)
# F0: The volume of a sphere is 4/3*pi*r**3 where r is the radius of the sphere. (Same as original fact F0)
A0=(4/3)*np.pi*(A3 ** 3) # (in**3)
""",
"""User:How much would the sea level rise if all the ice melted? (units: mi)

Assistant:
SUMMARY:= All of the ice on Earth has melted and flown into the ocean. We want to compute how much the sea level will rise.
PROGRAM:=
# Q0: How much would the sea level rise if all the ice melted?
# R0: This is the main question we want to answer.
# Q1: What is the volume of water that will be added to the ocean?
# R1: If we know the volume of water that will be added to the ocean, we can compute the sea level rise by dividing the volume of water by the surface area of the ocean.
# Q2: What is the volume of ice on Earth that is on land?
# R2: The volume of water that will be added to the ocean is the same as the volume of ice on Earth that is on land since that ice will melt and flow into the ocean.
# Q3: What is the volume of ice in Antarctica?
# R3: The majority of the ice on land is in Antarctica so it is a good approximation to use the volume of ice in Antarctica as the volume of ice on Earth that is on land.
# Q4: What is the thickness of the ice in Antarctica?
# R4: To compute the volume of ice in Antarctica, we need to know the thickness of the ice because we can multiply that by its surface area to get its volume.
# F4: The thickness of the ice in Antarctica is 4.8 km. We must convert this into mi by multiplying by 0.621371.
A4=0.621371 * 4.8e+0 # (mi)
# Q5: What is the surface area of Antarctica?
# R5: To compute the volume of ice in Antarctica, we need to know the surface area of Antarctica because we can multiply that by its thickness to get its volume.
# Q6: What fraction of the Earth's surface is Antarctica?
# R6: To compute the surface area of Antarctica, we need to know what fraction of the Earth's surface is Antarctica because we can multiply that by the surface area of the Earth to get the surface area of Antarctica.
# F6: Antarctica is 1/30th of the Earth's surface.
A6=1/30 # (dimensionless)
# Q7: What is the surface area of the Earth?
# R7: To compute the surface area of Antarctica, we need to know the surface area of the Earth because we can multiply that by the fraction of the Earth's surface that is Antarctica to get the surface area of Antarctica.
# F7: The surface area of the Earth is 1.97e+8 mi**2.
A7=1.97e+8 # (mi**2)
# F5: The surface area of Antarctica is the fraction of the Earth's surface that is Antarctica multiplied by the surface area of the Earth.
A5=A6*A7 # (mi**2)
# F3: The volume of ice in Antarctica is the surface area of Antarctica multiplied by the thickness of the ice in Antarctica.
A3=A5*A4 # (mi**3)
# F2: The volume of ice on Earth is approximated by the volume of ice in Antarctica.
A2=A3 # (mi**3)
# F1: The volume of water that will be added to the ocean is the same as the volume of ice on Earth.
A1=A2 # (mi**3)
# Q8: What is the surface area of the ocean?
# R8: To compute the sea level rise, we need to know the surface area of the ocean because we can divide the volume of water that will be added to the ocean by the surface area of the ocean to get the sea level rise.
# F8: The surface area of the ocean is 361 million km**2. We must convert this into mi**2 by multiplying by 0.386102.
A8=0.386102 * 361e+8 # (mi**2)
# F0: The sea level rise is the volume of water that will be added to the ocean divided by the surface area of the ocean.
A0=A1/A8 # (mi)

User:CONTEXT:=
- A basketball has a volume of 0.01 km**3.
- The mass of ice on land is 2.6e+19 kg.
- Antarctica is the largest content on Earth.
- A basketball has a volume of 0.01 km**3.
- The surface area of Antarctica is 5.4 million mi**2.
- The density of ice is 0.92 g/cm**3.
- The total volume of paint available to be painted on the wall is 0.13 cubic feet.
- An average human inhales 6 liters of air every minute

Assistant:
REVISED PROGRAM:=
# Q0: How much would the sea level rise if all the ice melted? (Same as original question Q0)
# R0: This is the main question we want to answer. (Same as original reasoning R0)
# Q1: What is the volume of water that will be added to the ocean? (Same as original question Q1)
# R1: If we know the volume of water that will be added to the ocean, we can compute the sea level rise by dividing the volume of water by the surface area of the ocean. (Same as original reasoning R1)
# Q2: What is the volume of ice on Earth that is on land? (Same as original question Q2)
# R2: The volume of water that will be added to the ocean is the same as the volume of ice on Earth that is on land since that ice will melt and flow into the ocean. (Same as original reasoning R2)
# Q3: What is the mass of the ice on Earth that is on land? (Revised question based on new context)
# R3: To compute the volume of ice on Earth that is on land, we need to know the mass of the ice because we can divide the mass of the ice by the density of ice to get the volume of the ice. (Revised reasoning based on new context)
# F3: The mass of ice on land is 2.6e+19 kg. (Revised fact based on new context)
A3=2.6e+19 # (kg)
# Q4: What is the density of ice? (Revised question based on new context)
# R4: To compute the volume of ice on Earth that is on land, we need to know the density of ice because we can divide the mass of the ice by the density of ice to get the volume of the ice. (Revised reasoning based on new context)
# F4: The density of ice is 0.92 g/cm**3. We must convert this into kg/(m**3) by multiplying by 1e+3. We must then convert this further into kg/(mi**3) by dividing by (6.2e-4 ** 3). (Revised fact based on new context)
A4=(1e+3)/(6.2e-4 ** 3) * 9.2e-1 # (kg/(mi**3))
# F2: The volume of ice on Earth that is on land is the mass of the ice on Earth that is on land divided by the density of ice. (Revised fact based on new context)
A2=A3/A4 # (mi**3)
# F1: The volume of water that will be added to the ocean is the same as the volume of ice on Earth that is on land. (Same as previous fact F1)
A1=A2 # (mi**3)
# Q5: What is the surface area of the ocean? (Same as original question Q8)
# R5: To compute the sea level rise, we need to know the surface area of the ocean because we can divide the volume of water that will be added to the ocean by the surface area of the ocean to get the sea level rise. (Same as original reasoning R8)
# F5: The surface area of the ocean is 361 million km**2. We must convert this into mi**2 by multiplying by 0.386102. (Same as original fact F8)
A5=0.386102 * 361e+8 # (mi**2)
# F0: The sea level rise is the volume of water that will be added to the ocean divided by the surface area of the ocean. (Same as original fact F0)
A0=A1/A5 # (mi)
"""
]