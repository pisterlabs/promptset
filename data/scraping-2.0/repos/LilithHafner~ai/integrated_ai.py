import openai
openai.api_key = "sk-..."

# GPT AI
def ai(prompt):
    response = openai.Completion.create(
      engine="code-davinci-002",
      prompt=prompt,
      temperature=0,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop="<end>"
    )
    return response.choices[0].text


# Subprocesses
def user(prompt):
    return input(prompt+"\n*>> ")

import traceback
def python_eval(prompt):
    try:
        return str(eval(prompt, globals()))
    except:
        return traceback.format_exc()
def python_exec(prompt):
    try:
        return str(exec(prompt, globals()))
    except:
        return traceback.format_exc()


subprocesses = [
    ("<user output>", "<user input>", user),
    ("<python eval>", "<python eval result>", python_eval),
    ("<python exec>", "<python exec result>", python_exec),
]
    

def subprocess(s):
    for start, end, func in subprocesses:
        if s.startswith(start):
            return end + func(s[len(start):])
#    print("The AI made an unsupported query:", s, "", sep="\n")
    return "<error>unknown tag"


## Training data
prompt = """This is a question and answer bot that has oracles to various external tools including python, google, and others

<user input>what time is it<end>
<pyhton eval>time.ctime()<end>
<python eval result>Traceback (most recent call last):
  File "/Users/x/Documents/integrated_ai.py", line 26, in python
    return str(eval(prompt, globals(), locals()))
  File "<string>", line 1, in <module>
NameError: name 'time' is not defined<end>
<python exec>import time<end>
<python exec result>None<end>
<python eval>time.ctime()<end>
<user output>The time is Sun Apr 24 18:01:32 2022<end>
<user input>what is the weather in New York<end>
<google>weather in New York<end>
<google result>Sunny
53°F°C
Precipitation: 1%
Humidity: 52%
Wind: 7 mph
New York, NY
Sunday 6:00 PM
Sunny
TemperaturePrecipitationWind<end>
<user output>The weather in New York is Sunny<end>
<user input>is it warm in chicago?<end>
<google>weather in chicago<end>
result: Cloudy
70°F°C
Precipitation: 5%
Humidity: 65%
Wind: 19 mph
Chicago, IL
Sunday 6:00 PM
Cloudy
TemperaturePrecipitationWind<end>
<user output>It is warm in chicago<end>
<user input>is 1729 prime?<end>
<python eval>is_prime(1729)<end>
<python eval result>Traceback (most recent call last):
  File "/Users/x/Documents/integrated_ai.py", line 26, in python_eval
    return str(eval(prompt, globals()))
  File "<string>", line 1, in <module>
NameError: name 'is_prime' is not defined<end>
<python exec>def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True<end>
<python exec result>None<end>
<python eval>is_prime(1729)<end>
<python eval result>False<end>
<user output>1729 is not prime<end>
<user input>Stop using google<end>
<user output>Google disabled.<end>
<user input>What's the weather?<end>
<user output>I cannot answer that question without google<end>
<user input>Name 7 edibe mushrooms<end>
<user output>Pleurotus, Lentinula edodes, Shiitake mushroom, Auricularia auricula-judae, Volvariella volvacea, Flammulina velutipes, Tremella fuciformis<end>"""

# Main loop
def kernal(verbose=True):
    global prompt
    prompt += "<user input>" + user("Welcome!") + "<end>\n"

    while True:
        call = ai(prompt)

        if verbose:
            print(call + "<end>")
        
        prompt += call + "<end>\n"

        if call.startswith("<exit>"):
            return

        result = subprocess(call)

        if verbose:
            print(result + "<end>")

        prompt += result + "<end>\n"
        

if __name__ == "__main__":
    kernal()
            
