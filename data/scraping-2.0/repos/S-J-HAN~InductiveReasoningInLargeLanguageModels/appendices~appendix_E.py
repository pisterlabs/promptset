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
Fact - Carrots have property P.
Claim - Rabbits have property P.

Argument B: 
Fact - Rabbits have property P.
Claim - Carrots have property P.
''',
'''
Argument A: 
Fact - Bananas have property P.
Claim - Monkeys have property P.

Argument B: 
Fact - Mice have property P.
Claim - Monkeys have property P.
''',
'''
Argument A: 
Fact - Fleas have property P.
Fact - Butterflies have property P.
Claim - Sparrows have property P.

Argument B: 
Fact - Fleas have property P.
Fact - Dogs have property P.
Claim - Sparrows have property P.
''',
'''
Argument A: 
Fact - Brown bears have property P.
Claim - Buffalos have property P.

Argument B: 
Fact - Brown bears have property P.
Fact - Polar bears have property P.
Fact - Grizzly bears have property P.
Claim - Buffalos have property P.
''',
'''
Argument A: 
Fact - Poodles can bite through wire.
Claim - German shepherds can bite through wire.

Argument B: 
Fact - Dobermanns can bite through wire.
Claim - German shepherds can bite through wire.
''',
'''
Argument A: 
Fact - House cats have skin that is more resistant to penetration than most synthetic fibers.
Claim - Hippos have skin that is more resistant to penetration than most synthetic fibers.

Argument B: 
Fact - Elephants have skin that is more resistant to penetration than most synthetic fibers.
Claim - Hippos have skin that is more resistant to penetration than most synthetic fibers.
''',
'''
Argument A: 
Fact - Chickens have livers with two chambers that act as one.
Claim - Hawks have livers with two chambers that act as one.

Argument B: 
Fact - Tigers have livers with two chambers that act as one.
Claim - Hawks have livers with two chambers that act as one.
''',
'''
Argument A: 
Fact - Tigers usually gather large amounts of food at once.
Claim - Hawks usually gather large amounts of food at once.

Argument B: 
Fact - Chickens usually gather large amounts of food at once.
Claim - Hawks usually gather large amounts of food at once.
''',
'''
Argument A: 
Fact - Ladybugs have some cells in their respiratory systems that require carbon dioxide to function.
Claim - Mosquitoes have some cells in their respiratory systems that require carbon dioxide to function.

Argument B: 
Fact - Vampire bats have some cells in their respiratory systems that require carbon dioxide to function.
Claim - Mosquitoes have some cells in their respiratory systems that require carbon dioxide to function.
''',
'''
Argument A: 
Fact - After eating, vampire bats travel at speeds of twice their body length per second.
Claim - After eating, mosquitoes travel at speeds of twice their body length per second.

Argument B: 
Fact - After eating, ladybugs travel at speeds of twice their body length per second.
Claim - After eating, mosquitoes travel at speeds of twice their body length per second.
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
