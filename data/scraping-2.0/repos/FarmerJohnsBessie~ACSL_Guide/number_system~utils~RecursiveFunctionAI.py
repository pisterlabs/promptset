from openai import OpenAI

client = OpenAI()
# ======================Files======================
recursion_file = client.files.create(
    file=open("number_system/utils/pdfs/Recursive Functions.pdf", "rb"),
    purpose='assistants'
)

pseudo_code_file = client.files.create(
    file=open("number_system/utils/pdfs/What does this program do.pdf", "rb"),
    purpose='assistants'
)

recursive_function_instruction_file = client.files.create(
    file=open("number_system/utils/pdfs/Recursive Function Instruction.txt", "rb"),
    purpose='assistants'
)

computer_number_system_file = client.files.create(
    file=open("number_system/utils/pdfs/Computer Number System.pdf", "rb"),
    purpose='assistants'
)

assistant = client.beta.assistants.create(
    name="ACSL Assistant",
    instructions='''You are a robot that can answer every questions from the ACSL competition. ACSL Competition is a 
    computer science competition. Therefore, you can only answer questions related to computer science. You can 
    answer questions related to the following topics: 
    1. Computer Number Systems (Question related to Binary, Octal, Decimal, Hexadecimal) 
    2. Recursion (Question related to recursive functions) 
    3. Psuedocode (Questions that ask what does this psuedocode do) 
    If you are not sure about the answer or not sure what to do, just output "I don't know" instead of trying to answer the question. 
    If a user asks a question that's not related to computer science, you MUST tell them that this is not related to ACSL and you can not answer this question.
    You have access to some files that contains instructions and examples to various ACSL topics. 
    These files can help you answer the questions. For example, the file "Recursive Functions.pdf" contains instructions and examples to recursive functions,
    and the file "What does this program do.pdf" contains the grammar and examples to psuedocode.
    
    Notice that you also have instruction files for each topic.
    For example, you have "Recursive Function Instruction.txt" for recursive functions.
    This file contains the instruction for recursive functions.
    You need to follow the instructions in the instruction file to answer the question.
    ''',
    tools=[{"type": "code_interpreter"}, {"type": "retrieval"}],
    model="gpt-4-1106-preview",
    file_ids=[recursion_file.id, pseudo_code_file.id, recursive_function_instruction_file.id, computer_number_system_file.id]
)


thread = client.beta.threads.create()

# recursion_statement = ['3*x+2', 'f(x-1)+4']
# recursion_conditional = ['x<3', 'x>=3']

def recursive_function_solver(recursion_statement, recursion_conditional, value):
    recursion_solving_prompt = '''
Read the instructions in Recursive Function Instruction.txt
Please solve the recursion question below.
Output without any steps, just the numerical answer. Follow the answer format.

Define f(x)/f(x,y) = 

'''
    for i in range(len(recursion_statement)):
        line = f"{recursion_statement[i]} if {recursion_conditional[i]}"
        recursion_solving_prompt += line + "\n"

    recursion_solving_prompt += f'''
    Problem Statement:
    Evaluate f({value}).
    '''
    print(recursion_solving_prompt)

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=recursion_solving_prompt
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions=""
    )
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        if run.status == "completed":
            break
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    first_message_value = messages.data[0].content[0].text.value
    return first_message_value
