from termcolor import colored, cprint
import os
import openai
import sys


cprint("PSA: idk maybe you should only use this for people you've been casually, informally dating (<1 month of dating) but thats just our opinion", "black", "on_red")
print('\n')
cprint("Tell us about the other person [Type your response]", "red", "on_black")
name = input(colored("What is his/her/their name? ", "magenta", "on_black"))
sign = input(colored("What is his/her/their horoscope sign? (optional) ", "magenta", "on_black"))
reason = input(colored("Why is it not working out? (optional) ", "magenta", "on_black"))
friends = input(colored("Do you want to extend the option to remain friends with them? (Yes/No) ", "magenta", "on_black"))

print('\n')
apiKey = input(colored("Please enter a valid OpenAI API key: ", "green", "on_black"))
print('\n')

cprint("LOADING BREAKUP LETTER ", "red", "on_black", attrs=["blink"])

story = ""

if not(apiKey): 
    cprint("An OpenAI API key is required")
    sys.exit()    
if not(name): 
    cprint("Name is required")
    sys.exit()    
else: 
    story += f"Their name is {name.strip()}. "
if sign:
    story += f"Their star sign is {sign.strip()}. I think the breakup letter should take into account the stereotypical emotional state of this sign in the approach of the message, but doesn't mention their star sign. "
if reason:
    story += f"I am no longer interested in dating them because of the following reason: {reason.strip()}. "
if friends:
    if friends.lower().strip() == "yes":
        story += f"I want to extend the option to remain friends with them in the future. "
    elif friends.lower().strip()=="no":
        story += f"I do not want to extend the option to remain friends with them in the future. "
    else:
        cprint("Please respond YES or NO when asked if you would like to remain friends with them in the future. ")
        sys.exit()    

openai.api_key = apiKey

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a close helpful friend trying to help your friend navigate a breakup."},
    {"role": "user", "content": "Help me write a very short gen-z style text message that is 3 lines text to send in order to break up with someone. Don't include their name. " + f"{story}"}
  ]
)


print('\n')
cprint(completion.choices[0].message["content"], "white", "on_black", attrs=["bold"])



