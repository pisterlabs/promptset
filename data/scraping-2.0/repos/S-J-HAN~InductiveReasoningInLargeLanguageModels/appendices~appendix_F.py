import openai
import json 

openai.api_key = "APIKEY"

system_message = '''
You will be asked a series of questions that don't have right or wrong answers. You are willing to use your best judgment and commit to a concrete, specific response even in cases where you can't be sure that you are correct.
'''

questions= [
'''
Animal A has scales and can fly. How likely is it that Animal A has wings?
''',
'''
Animal A has scales and two legs. How likely is it that Animal A has wings?
''',
'''
Do you think they might grow rice in Florida?
''',
'''
Can a goose quack?
''',
'''
I have a circle-shaped object that is 3 inches in diameter. Do you think it's more likely to be a pizza or a quarter?
''',
'''
You are at a party and you learn that one of the guests fell into the pool. Why do you think this happened?
''',
'''
The doctors took a raccoon and shaved away some of its fur. They dyed what was left all black. Then they bleached a single stripe all white down the center of its back. Then, with surgery, they put in its body a sac of super smelly odor, just like a skunk has. When they were done, the animal looked just like a skunk. After the operation was this a skunk or a raccoon?
''',
'''
The doctors took a coffeepot. They sawed off the handle, sealed the top, took off the top knob, sealed closed the spout, and sawed it off. They also sawed off the base and attached a flat piece of metal. They attached a little stick, cut a window in it, and filled the metal container with bird food. After the operation was this a coffeepot or a bird feeder?
''',
]

suffix= '''
Please explain your answer carefully.
'''



for question in questions:
    user_message =  question + suffix
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
