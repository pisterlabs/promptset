import openai
import json
from envs import chatGPT_API_Key

openai.api_key = chatGPT_API_Key

def chat_with_gpt3(prompt):
    response = openai.Completion.create(
        engine='text-davinci-002',  # Choose the ChatGPT model you prefer
        prompt=prompt,
        max_tokens=1000,  # Set the maximum length of the response
        temperature=0,  # Controls the randomness of the response
        n=1,  # Set the number of responses to generate
        stop=None  # Specify a stop token if desired
    )
    
    return response.choices[0].text.strip()

def gpt2objects(instring):
    return json.loads(instring)

def gptFromTopic(textIn):
    prompt = """In our previous conversation, I asked you to break down my text into objects representing individual claims of an argument, with each claim linked to previous claims if the evidence in the former justifies the latter.  There may be a lot of fluff, but I want you to simplify the text to *just* the claims.  Also, respond *only* with the objects.  Do not say anything else.

This time, I want you to create the objects (~15 sounds good) from the following topic:
{textIn}

Please format these objects as follows:
{"objects": 
[{"id": 1, "parent": "None", "text": "Text of the first claim."},
{"id": 2, "parent": 1, "text": "Text of the second claim that justifies the first claim."},
{...}]
}

where 'parent' is NOT the id of the previous claim but of the broader claim that this new claim references.
Limit the depth of any oject chain to ~4.  
You're limited to a maximum of 1000 tokens, so do not exceed that.

Thank you!""".replace("{textIn}", textIn)
    return chat_with_gpt3(prompt)


def gptFromPreexisting(textIn, id):
    prompt = """In our previous conversation, I asked you to break down my text into objects representing individual claims of an argument, with each claim linked to previous claims if the evidence in the former justifies the latter.  There may be a lot of fluff, but I want you to simplify the text to *just* the claims.  Also, respond *only* with the objects.  Do not say anything else.

This time I want you to generate objects that disprove the claim below (it's id is {id}):
{textIn}

Please format these objects as follows:
{"objects": 
[{"id": 1, "parent": "None", "text": "Text of the first claim."},
{"id": 2, "parent": 1, "text": "Text of the second claim that justifies the first claim."},
{...}]
}

where 'parent' is NOT the id of the previous claim but of the broader claim that this new claim references.
Limit the depth of any oject chain to ~4.  
You're limited to a maximum of 1000 tokens, so do not exceed that.

Thank you!""".replace("{id}", id).replace("{textIn}", textIn)
    return chat_with_gpt3(prompt)


def gptFromArgs(textIn):
    prompt = """
In our previous conversation, I asked you to break down my text into objects representing individual claims of an argument, with each claim linked to previous claims if the evidence in the former justifies the latter.  There may be a lot of fluff, but I want you to simplify the text to *just* the claims.  Also, respond *only* with the objects.  Do not say anything else.

Here's the text I'd like you to process into these objects:
{textIn}

Please format these objects as follows:
{"objects": 
[{"id": 1, "parent": "None", "text": "Text of the first claim."},
{"id": 2, "parent": 1, "text": "Text of the second claim that justifies the first claim."},
{...}]
}

where 'parent' is NOT the id of the previous claim but of the broader claim that this new claim references.
Limit the depth of any oject chain to ~4.  
You're limited to a maximum of 1000 tokens, so do not exceed that.

Thank you!
""".replace("{textIn}", textIn)
    return chat_with_gpt3(prompt)

# Limit to 300 words or 2000 characters
# print(gptFromArgs("""
# If you are wondering who will do my Python homework, don't worry because you can always find a professional homework help service online. On specialized websites, experts can provide students with Python homework help online of any complexity.

# Website helpers provide students their services at a low price, no matter where you study, at school, college or university. Professionals will deliver a quality solution without plagiarism according to the established deadline. You can choose the deadline that suits you best. Your home assignment solution can be delivered in 5 hours or 5 days.

# Do not worry about the quality of work, because each helper has a degree and has graduated from a higher education institution. You can even check samples of how helpers did homework of a certain type for other students. All the experts have a long experience in helping students with homework online.

# The most important thing is that you can seek assistance at any time. You can order a homework solution even at night if you suddenly forget about it.
# Which Programming Language Should I Start Learning?

# If you like a STEM subject, and you have decided that you want to connect your career with IT, you need to know which of the programming languages will help you in your work. Most likely, you will start learning the programming language with Python.

# Python is one of the simplest programming languages that is becoming more popular in the IT world. Python ranks fourth in popularity among programming languages, second only to the classic Java, C, and C ++. Today, every programmer must own this tool to be successful in his niche.

# Python is a multipurpose programming language that allows programmers to write code that reads well. The relative conciseness of Python allows you to create a program that will be much shorter than its counterpart written in another language.
# """))

# print(gptFromTopic("effectiveness of electric vehicles"))
