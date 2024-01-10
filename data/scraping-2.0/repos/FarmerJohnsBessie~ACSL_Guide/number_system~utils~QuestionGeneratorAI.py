from openai import OpenAI

try:
    client = OpenAI()
    thread = client.beta.threads.create()
except Exception as e:
    pass


def generate_question(question_type, difficulty, additional_prompt):
    files = {
        'Computer_Number_Systems': 'file-CQ7N9RnwoPQMYup3kZSYgtQF',
        'Recursive_Functions': 'file-9rA4i6x6ZOTP0ffbDvVo9foO',
        'What_Does_This_Program_Do': 'file-N65Qf7gOUYBuUAjyDH51NsnC'
    }
    language = {
        'Computer_Number_Systems': 'computer number systems',
        'Recursive_Functions': 'revcursive functions',
        'What_Does_This_Program_Do': 'ACSL pseudocode'
    }

    sample_outputs = {
        'Recursive_Functions': r'''
    Problem:
    Consider the recursive function \( W(x) \) defined by the following rules:
    
    \[ W(x) = \begin{cases} 
    3x + 1 & \text{if } x \leq 5 \\
    W(x - 2) - 1 & \text{if } x > 5 
    \end{cases} \]
    
    Calculate the value of \( W(9) \).
    
    Steps:
    \[ W(9) = W(7) - 1 \]
    \[ W(7) = W(5) - 1 \]
    \[ W(5) = 3 \times 5 + 1 = 16 \]
    Substitute \( W(5) \) back into the previous expression:
    \[ W(7) = 16 - 1 = 15 \]
    Substitute \( W(7) \) back into the initial expression:
    \[ W(9) = 15 - 1 = 14 \]
    Therefore, the value of \( W(9) \) is \( 14 \).
    
    Answer:
    14
            ''',
            'Computer_Number_Systems': r'''
    Problem:
    Solve for x where \(x_{16}\)=\(3676_{8}\)
    
    Steps:
    One method of solution is to convert \(3676_{8}\) into base 10, and then convert that number into base 16 to yield the value of x.
    
    An easier solution, less prone to arithmetic mistakes, is to convert from octal (base 8) to hexadecimal (base 16) through the binary (base 2) representation of the number:
    
    \[ 3676_{8} = 011 110 111 110_{2} \]
    \[ 011 110 111 110_{2} = 7BE_{16} \]
    Answer:
    \(7BE_{16}\)
        ''',
        'What_Does_This_Program_Do': 'sort'
    }

    addtional_comments = {
        'Recursive_Functions': r'''
You need to write your question in math expressions. You need to the mathjax format to write math expressions.
For example, you need to express an recursive function like this:
\[ W(x) = \begin{cases} 
3x + 1 & \text{if } x \leq 5 \\
W(x - 2) - 1 & \text{if } x > 5 
\end{cases} \]
''',
        'Computer_Number_Systems': r'''
''',
        'What_Does_This_Program_Do': r'''
'''
    }
    example_recursion_questions = r'''
    Here are some actual recursion questions from the ACSL competition:
    The problems are very creative and unique. 
    Your generated problems should have similar difficulties with these problems.
    You should not just copy these problems, or just changes some numbers you need to generate your own problems.
    You can use these problems as a reference to generate your own problems.
    1. Find \( g(11) \) given the following
    \[ g(x) = \begin{cases} g(x - 3) + 1 & \text{if } x > 0 \\ 3x & \text{otherwise} \end{cases} \]
    Difficulty: Easy

    2. Find the value of \( h(13) \) given the following definition of \( h \):
    \[ h(x) = \begin{cases} h(x - 7) + 1 & \text{when } x > 5 \\ x & \text{when } 0 \leq x \leq 5 \\ h(x + 3) & \text{when } x < 0 \end{cases} \]
    Difficulty: Medium

    3. Problem: Find the value of \( f(12, 6) \) given the following definition of \( f \):
    \[ f(x, y) = \begin{cases} f(x - y, y - 1) + 2 & \text{when } x > y \\ x + y & \text{otherwise} \end{cases} \]
    Difficulty: Hard

    4. Problem: Consider the following recursive algorithm for painting a square:
    1. Given a square.
    2. If the length of a side is less than 2 feet, then stop.
    3. Divide the square into 4 equal-size squares (i.e., draw a “plus” sign inside the square).
    4. Paint one of these 4 small squares.
    5. Repeat this procedure (start at step 1) for each of the 3 unpainted squares.
    If this algorithm is applied to a square with a side of 16 feet (having a total area of 256 sq. feet), how many square feet will be painted?
    Difficulty: Hard

    5. Find \( f(12) \) given:
    \[ f(x) = \begin{cases} f(x-2) - 3 & \text{if } x \geq 10 \\ f(2x-10) + 4 & \text{if } 3 \leq x < 10 \\ x^2 + 5 & \text{if } x < 3 \end{cases} \]
    Difficulty: Medium

    6. Find \( f(15) \) given:
    \[ f(x) = \begin{cases} 2{f(x-3)}-4 & \text{if } x \geq 10 \\ f(x+1) + f(x-2) & \text{if } 8 \leq x < 10 \\ f(x-4) - 1 & \text{if } 5 \leq x < 8 \\ x + 3 & \text{if } x < 5 \end{cases} \]
    Difficulty: Hard

    7. Find \( f(15,12) \) given:
    \[ f(x, y) = \begin{cases} f(y-1,x-2) + 4 & \text{if } x > 10 \\ f(x+3,y-3) + 2 & \text{if } 5 \leq x \leq 10 \\ 3x - 2y & \text{if } x \leq 4 \end{cases} \]
    Difficulty: Hard

    8. Find \( f(10) \), given:
    \[ f(x) = \begin{cases} 1 & \text{if } x = 1 \\ f(x-1) + 3x - 2 & \text{if } x > 1 \end{cases} \]
    Difficulty: Medium

    9. Evaluate \( f(100, 36) - f(36, 100) \), given:
    \[ f(x,y) = \begin{cases} f\left(\left[\frac{x}{2}\right], \left[\frac{y}{2}\right]\right) + 2 & \text{if } xy \geq 50 \\ f(2x, y-3) + 1 & \text{if } 10 < xy < 50 \\ xy - x - y & \text{if } xy \leq 10 \end{cases} \]
    where \( [z] \) returns the greatest integer less than or equal to \( z \).
    Difficulty: Insane
    '''

    prompt = f'''
    read the PDF file {files[question_type]} if possible and create a unique {difficulty} {language[question_type]} question that has the same style. 
    Same style also means the the format of math expressions.
    Based on the style and format of the provided examples and sample problems from the PDF file, {difficulty} {language[question_type]} question.

    Also output the steps you took to solve the question. (basically the math you did).
    
    The problem you generated should be creative.
    The problem you generated shouldn't involve hard and nasty calculations. (like 123456789 * 987654321) 
    The problem you generated should have a single numerical answer. (not multiple parts) 
    
    This is a sample output for a {language[question_type]} question:
    {example_recursion_questions}
    
    {additional_prompt}

    output math in mathjax formula.
    follow the output format strictly, do not change the format, do not output any other things.
    you need to output both the problem, steps and answer in the correct format in your last response.
    DO NOT OUTPUT ANYTHING THAT DOESN't FOLLOW THE OUTPUT FORMAT!!!
    DO NOT OUTPUT ANY FILLER WORDS AT THE BEGINNING OR THE END OF THE PROBLEM!
    '''

    prompt2= f'''
    You are asked to generate a {difficulty} {language[question_type]} question.
    Here are the steps you need to follow:
    1. Read the PDF file {files[question_type]} and look at the example problems inside it. The problem you generated 
    should have the same style as the example problems. By same style, I mean the general format of the problem, 
    like the wording of the question, the math expression for the question or the answer, etc. Same style also means 
    the the format of math expressions. 
    2. Make sure you understand the difficulty of the problem you are generating. (which is {difficulty}) 
    3. Make sure that this problem is unique. (not the same as any of the example problems) 
    4. The problem you generated should be creative. 
    5. The problem you generated should be solvable. 
    6. The problem you generated shouldn't involve hard and nasty calculations. (like 123456789 * 987654321) 
    7. The problem you generated should have a single numerical answer. (not multiple parts) 
    8. If there are many math expressions in the problem, make sure that the math expressions are in the mathjax format. 
    9. Make sure the problem generated is in the correct format. Correct Format (only to show you the format, not the actual problem): 
    Problem: 1+1 
    Steps: 1+1=2 
    Answer: 2 
    10. Make sure the problem generated is in the correct format. Do not output anything that doesn't 
    follow the format. 
    11. Do not output any additional text that is not the part of the format. 
    12. Now, output the question you generated in the correct format. You need to output both the problem, steps and 
    answer in the correct format in your last response. 
    
    Additional Comments:
    For {language[question_type]} questions:
    {addtional_comments[question_type]}
    
    {example_recursion_questions}
    
    DO NOT OUTPUT ANYTHING THAT DOESN't FOLLOW THE OUTPUT FORMAT!!!
    '''



    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id="asst_7FtHyXIwL5vSlsDgBHBaMyPn",
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

