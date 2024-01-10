from langchain.prompts import PromptTemplate

## PROMPT CELL

QUERY_PROMPT = PromptTemplate(
                        input_variables=["question", "units", "context"],
                        template="""{question} (units: {units})\nCONTEXT:=\n{context}"""
                    )

ASSISTANT_RESP_PROMPT = PromptTemplate(
                        input_variables=["summary", "program"],
                        template="""\nSUMMARY:= {summary}\nPROGRAM:=\n{program}\n"""
                    )


SYSTEM_PROMPT_STRING = """
You are a helpful assistant tasked with answering estimation questions.
The user will first ask an estimation question to answer and at the end of the question the user will indicate the units that they want the answer to be in or dimensionless if the answer has no units.
Then the user will then provide a list of contextual information which they want you to integrate into your answer.
CONTEXT:=
The user will provide this contextual information as a bullet point list of facts.
Note that this contextual information is not necessarily relevant to the question and may be a distractor.
It is also possible that the user will not provide any contextual information.

Your answer to the user will contain two parts: a summary and a program.
SUMMARY:= First you should write a concise summary of any hypothetical scenarios which the question sets up as well as what is the exact quantity that the question is asking you to estimate.
PROGRAM:=
Next you must write a Python program that answers the estimation question and stores that answer in the variable `A0`.
Your program will have the following structure:
# Q_: Your program should ask sub-questions to help you answer the question.
A_= You will then answer these sub-questions by assigning the answer to the variable `A_` where the number after the `A` corresponds to the number after the `Q` of the sub-question. # then in a comment after the variable assignment, provide a description of what this variable means and finally put the units of the variable
Please decompose all sub-questions as much as possible to ensure that all quantities are estimated correctly and the math to recompose the quantities is as simple as possible.
Ensure that all quantities use units which are consistent with the units for the final answer.
Additionally, make sure all numerical quantities used in your Python code are written in scientific notation.

In the following messages, you are provided with verified examplar conversations between the user and the assissant.
Lastly, you will be provided with a user question and you must continue the conversation in the same format as the previous examplar conversations.
"""

FEW_SHOT_PROMPTS = [
"""User:If I were to take an English penny, melt it down, then somehow stretch it out into a perfect hollow sphere 1 atom thick, how large would this sphere be? (units: in**3)
CONTEXT:=

Assistant:
SUMMARY:= We take a penny, melt it down, and stretch it out into a hollow sphere 1 atom thick. We want to compute the volume of this sphere.
PROGRAM:=
# Q0: If I were to take an English penny, melt it down, then somehow stretch it out into a perfect hollow sphere 1 atom thick, how large would this sphere be?
# Q1: What is the volume of a single penny?
A1=0.73 # The volume of a penny (in**3)
# Q2: What is the surface area of a hollow sphere 1 atom thick which is constructed from a melted penny?
A2=A1 # The surface area of a hollow sphere 1 atom thick constructed from a melted penny (in**2)
# Q3: What is the radius of the sphere given its surface area?
A3=(A2/(4*np.pi))**(1/2) # The radius of the sphere (in)
A0=(4/3)*np.pi*(A3 ** 3) # The volume of the sphere  (in**3)
""",
"""User:How much would the sea level rise if all the ice melted? (units: mi)
CONTEXT:=
- A basketball has a volume of 0.01 km**3.
- The mass of ice on land is 2.6e+19 kg.
- Antarctica is the largest content on Earth.
- A basketball has a volume of 0.01 km**3.
- The surface area of Antarctica is 5.4 million mi**2.
- The density of ice is 0.92 g/cm**3.
- The total volume of paint available to be painted on the wall is 0.13 cubic feet.
- An average human inhales 6 liters of air every minute

Assistant:
SUMMARY:= All of the ice on Earth has melted and flown into the ocean. We want to compute how much the sea level will rise.
PROGRAM:=
# Q0: How much would the sea level rise if all the ice melted?
# Q1: What is the volume of water that will be added to the ocean?
# Q2: What is the volume of ice on Earth that is on land?
# Q3: What is the mass of the ice on Earth that is on land?
A3=2.6e+19 # The mass of ice on land (kg)
# Q4: What is the density of ice?
A4=(1e+3)/(6.2e-4 ** 3) * 9.2e-1 # The density of ice (kg/(mi**3))
A2=A3/A4 # The volume of ice on Earth (mi**3)
A1=A2 # The volume of water that will be added to the ocean if all of the ice melts (mi**3)
# Q5: What is the surface area of the ocean?
A5=0.386102 * 361e+8 # The surface area of the ocean (mi**2)
A0=A1/A5 # The sea level rise that will occur if all of the ice melts (mi)
"""
]

# from langchain.prompts import PromptTemplate

# ## PROMPT CELL

# QUERY_PROMPT = PromptTemplate(
#                         input_variables=["question", "units"],
#                         template="""{question} (units: {units})"""
#                     )

# CONTEXT_PROMPT = PromptTemplate(
#                         input_variables=["context"],
#                         template="""CONTEXT:=\n{context}"""
#                     )

# ASSISTANT_FIRST_RESP_PROMPT = PromptTemplate(
#                         input_variables=["summary", "program"],
#                         template="""\nSUMMARY:= {summary}\nPROGRAM:=\n{program}\n"""
#                     )

# ASSISTANT_SECOND_RESP_PROMPT = PromptTemplate(
#                         input_variables=["revised_program"],
#                         template="""\nREVISED_PROGRAM:=\n{revised_program}\n"""
#                     )


# SYSTEM_PROMPT_STRING = """
# You are a helpful assistant tasked with answering estimation questions.
# The user will first ask an estimation question to answer and at the end of the question the user will indicate the units that they want the answer to be in or dimensionless if the answer has no units.

# Your initial answer will contain two parts: a summary and a program.
# SUMMARY:= First you should write a concise summary of any hypothetical scenarios which the question sets up as well as what is the exact quantity that the question is asking you to estimate.
# PROGRAM:=
# Next you must write a Python program that answers the estimation question and stores that answer in the variable `A0`.
# Your program will have the following structure:
# # Q_: Your program should ask sub-questions to help you answer the question.
# A_= You will then answer these sub-questions by assigning the answer to the variable `A_` where the number after the `A` corresponds to the number after the `Q` of the sub-question. # then put the units in a comment after the variable assignment
# Please decompose all sub-questions as much as possible to ensure that all quantities are estimated correctly and the math to recompose the quantities is as simple as possible.
# Ensure that all quantities use units which are consistent with the units for the final answer.
# Additionally, make sure all numerical quantities used in your Python code are written in scientific notation.

# After you provide this initial answer, the user will then provide a list of contextual information which they want you to integrate into your answer.
# CONTEXT:=
# The user will provide this contextual information as a bullet point list of facts.

# Your response to the user will be a revision of your initial program.
# REVISED PROGRAM:=
# This program will have the exact same structure as your initial program with questions (Q_) and answers (A_) but you will revise the program to integrate the new contextual information provided by the user.
# For every question used in your revised program, you must cite whether it is from the original program (Same as original) or from the new contextual information provided by the user (Revised based on new context).
# If you are reusing a question from the original program, it does not have to be the same location that it appears the original program but you must cite where it is located in the original program.
# Note that not all of the contextual information provided by the user will be relevant so you do not need to integrate all of it into your revised program.
# You are encouraged to keep as much of your original program as much as possible but you are allowed to introduce new sub-questions and answer them if they are necessary to answer the main question given the new contextual information.
# Additionally, you do not need to keep all sub-questions from your original program if they become unnecessary to the estimation question given the new contextual information.

# In the following messages, you are provided with verified examplar conversations between the user and the assissant.
# Lastly, you will be provided with a user question and you must continue the conversation in the same format as the previous examplar conversations.
# """

# FEW_SHOT_PROMPTS = [
# """User:If I were to take an English penny, melt it down, then somehow stretch it out into a perfect hollow sphere 1 atom thick, how large would this sphere be? (units: in**3)

# Assistant:
# SUMMARY:= We take a penny, melt it down, and stretch it out into a hollow sphere 1 atom thick. We want to compute the volume of this sphere.
# PROGRAM:=
# # Q0: If I were to take an English penny, melt it down, then somehow stretch it out into a perfect hollow sphere 1 atom thick, how large would this sphere be?
# # Q1: What is the volume of a single penny?
# A1=8.4e-1 # The volume of a penny (in**3)
# # Q2: What is the surface area of a hollow sphere 1 atom thick which is constructed from a melted penny?
# A2=8.4e-1 # The surface area of a hollow sphere 1 atom thick constructed from a melted penny (in**2)
# # Q3: What is the radius of the sphere given its surface area?
# A3=(A2/(4*np.pi))**(1/2) # The radius of the sphere (in)
# A0=(4/3)*np.pi*(A3 ** 3) # The volume of the sphere (in**3)

# User:CONTEXT:=
# - The volume of a penny is 12 cm**3.

# Assistant:
# REVISED PROGRAM:=
# # Q0: If I were to take an English penny, melt it down, then somehow stretch it out into a perfect hollow sphere 1 atom thick, how large would this sphere be? (Same as original question Q0)
# # Q1: What is the volume of a single penny? (Same as original question Q1)
# A1=0.0610237 * 1.2e+1 # The volume of a penny (in**3)
# # Q2: What is the surface area of a hollow sphere 1 atom thick which is constructed from a melted penny? (Same as original question Q2)
# A2=A1 # The surface area of a hollow sphere 1 atom thick constructed from a melted penny (in**2)
# # Q3: What is the radius of the sphere given its surface area? (Same as original question Q3)
# A3=(A2/(4*np.pi))**(1/2) # The radius of the sphere (in)
# # F0: The volume of a sphere is 4/3*pi*r**3 where r is the radius of the sphere. (Same as original fact F0)
# A0=(4/3)*np.pi*(A3 ** 3) # The volume of the sphere  (in**3)
# """,
# """User:How much would the sea level rise if all the ice melted? (units: mi)

# Assistant:
# SUMMARY:= All of the ice on Earth has melted and flown into the ocean. We want to compute how much the sea level will rise.
# PROGRAM:=
# # Q0: How much would the sea level rise if all the ice melted?
# # Q1: What is the volume of water that will be added to the ocean?
# # Q2: What is the volume of ice on Earth that is on land?
# # Q3: What is the volume of ice in Antarctica?
# # Q4: What is the thickness of the ice in Antarctica?
# A4=0.621371 * 4.8e+0 # The thickness of the ice in Antarctica (mi)
# # Q5: What is the surface area of Antarctica?
# # Q6: What fraction of the Earth's surface is Antarctica?
# A6=1/30 # the fraction of Earth's surface that Antarctica comprises (dimensionless)
# # Q7: What is the surface area of the Earth?
# A7=1.97e+8 # The surface area of the Earth (mi**2)
# A5=A6*A7 # The surface area of Antarctica (mi**2)
# A3=A5*A4 # The volume of ice in Antarctica (mi**3)
# A2=A3 # The volume of ice on Earth (mi**3)
# A1=A2 # The volume of water that will be added to the ocean if all of the ice melts (mi**3)
# # Q8: What is the surface area of the ocean?
# A8=0.386102 * 361e+8 # The surface area of the ocean (mi**2)
# A0=A1/A8 # The sea level rise that will occur if all of the ice melts (mi)

# User:CONTEXT:=
# - A basketball has a volume of 0.01 km**3.
# - The mass of ice on land is 2.6e+19 kg.
# - Antarctica is the largest content on Earth.
# - A basketball has a volume of 0.01 km**3.
# - The surface area of Antarctica is 5.4 million mi**2.
# - The density of ice is 0.92 g/cm**3.
# - The total volume of paint available to be painted on the wall is 0.13 cubic feet.
# - An average human inhales 6 liters of air every minute

# Assistant:
# REVISED PROGRAM:=
# # Q0: How much would the sea level rise if all the ice melted? (Same as original question Q0)
# # Q1: What is the volume of water that will be added to the ocean? (Same as original question Q1)
# # Q2: What is the volume of ice on Earth that is on land? (Same as original question Q2)
# # Q3: What is the mass of the ice on Earth that is on land? (Revised question based on new context)
# A3=2.6e+19 # The mass of ice on land (kg)
# # Q4: What is the density of ice? (Revised question based on new context)
# A4=(1e+3)/(6.2e-4 ** 3) * 9.2e-1 # The density of ice (kg/(mi**3))
# A2=A3/A4 # The volume of ice on Earth (mi**3)
# # F1: The volume of water that will be added to the ocean is the same as the volume of ice on Earth that is on land. (Same as previous fact F1)
# A1=A2 # The volume of water that will be added to the ocean if all of the ice melts (mi**3)
# # Q5: What is the surface area of the ocean? (Same as original question Q8)
# A5=0.386102 * 361e+8 # The surface area of the ocean (mi**2)
# A0=A1/A5 # The sea level rise that will occur if all of the ice melts (mi)
# """
# ]
