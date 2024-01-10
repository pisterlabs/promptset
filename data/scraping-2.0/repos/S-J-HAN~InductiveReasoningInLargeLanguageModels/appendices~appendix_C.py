import openai
import json 

openai.api_key = "APIKEY"

system_message = '''
You are an expert on animals and the types of real world properties that they have. The questions you'll see don't have right or wrong answers, and you are willing to use your best judgment and commit to a concrete, specific response even in cases where you can't be sure that you are correct.
'''

context = '''
We are interested in how people evaluate arguments. On each trial there will be two arguments labeled 'A' and 'B.' Each will contain one, two, or three statements separated from a claim by a line. Assume that the statements above the line are facts, and choose the argument whose facts provide a better reason for believing the claim. These are subjective judgments; there are no right or wrong answers.
'''

question = '''
Question: Assuming all the facts given are true, which argument makes a stronger case for the claim? To get the best answer, first write down your reasoning. Then, based on this, indicate your preference by providing one of the following options:
A - Argument A is much stronger
B - Argument A is moderately stronger
C - Argument A is slightly stronger
D - Argument B is slightly stronger
E - Argument B is moderately stronger
F - Argument B is much stronger
'''

arguments = [
'''
Argument A: 
Fact - Robins have property P.
Fact - Bluejays have property P.
Claim - Sparrows have property P.

Argument B: 
Fact - Robins have property P.
Fact - Bluejays have property P.
Claim - Geese have property P.
''',
'''
Argument A: 
Fact - Robins have property P.
Claim - All birds have property P.

Argument B: 
Fact - Penguins have property P.
Claim - All birds have property P.
''',
'''
Argument A: 
Fact - Bluejays have property P.
Fact - Falcons have property P.
Claim - All birds have property P.

Argument B: 
Fact - Bluejays have property P.
Fact - Falcons have property P.
Claim - All animals have property P.
''',
'''
Argument A: 
Fact - Sparrows have property P.
Fact - Eagles have property P.
Fact - Hawks have property P.
Claim - All birds have property P.

Argument B: 
Fact - Sparrows have property P.
Fact - Eagles have property P.
Claim- All birds have property P.
''',
'''
Argument A: 
Fact - Pigs have property P.
Fact - Wolves have property P.
Fact - Foxes have property P.
Claim - Gorillas have property P.

Argument B: 
Fact - Pigs have property P.
Fact - Wolves have property P.
Claim - Gorillas have property P.
''',
'''
Argument A: 
Fact - Hippos have property P.
Fact - Hamsters have property P.
Claim - All mammals have property P.

Argument B: 
Fact - Hippos have property P.
Fact - Rhinos have property P.
Claim - All mammals have property P.
''',
'''
Argument A: 
Fact - Lions have property P.
Fact - Giraffes have property P.
Claim - Rabbits have property P.

Argument B: 
Fact - Lions have property P.
Fact - Tigers have property P.
Claim - Rabbits have property P.
''',
'''
Argument A: 
Fact - Crows have property P.
Fact - Peacocks have property P.
Claim - All birds have property P.

Argument B: 
Fact - Crows have property P.
Fact - Peacocks have property P.
Fact - Rabbits have property P.
Claim - All birds have property P.
''',
'''
Argument A: 
Fact - Flies have property P.
Claim - Bees have property P.

Argument B: 
Fact - Flies have property P.
Fact - Orangutans have property P.
Claim - Bees have property P.
''',
'''
Argument A: 
Fact - Mice have property P.
Claim - Bats have property P.

Argument B: 
Fact - Bats have property P.
Claim - Mice have property P.
''',
'''
Argument A: 
Fact - Robins have property P.
Claim - All birds have property P.

Argument B: 
Fact - Robins have property P.
Claim - Ostriches have property P.
''',
]

for argument in arguments:
    user_message = context + argument + question
    prompt3 = system_message + user_message
    prompt4 = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

    output3= openai.Completion.create(
     model="text-davinci-003",   
     prompt=prompt3,
     max_tokens=512,
     temperature=0,
    )
    gpt3_completion_text = output3.choices[0].text

    with open('gpt3_run_output_fresh.txt', 'a') as file:
        file.write(gpt3_completion_text + '\n****************\n')

    with open('gpt3_run_full_output_fresh.txt', 'a') as file:
        json.dump(output3, file)
        file.write('\n')

    with open('gpt3_run_input_fresh.txt', 'a') as file:
        file.write(prompt3 + '\n****************\n')

    output4= openai.ChatCompletion.create(
     model="gpt-4-0314",
     messages=prompt4,
     temperature=0,
    )
    gpt4_completion_text = output4.choices[0].message.content

    with open('gpt4_run_output_fresh.txt', 'a') as file:
        file.write(gpt4_completion_text + '\n****************\n')

    with open('gpt4_run_full_output_fresh.txt', 'a') as file:
        json.dump(output4, file)
        file.write('\n')

    with open('gpt4_run_input_fresh.txt', 'a') as file:
        json.dump(prompt4, file)
        file.write('\n')
