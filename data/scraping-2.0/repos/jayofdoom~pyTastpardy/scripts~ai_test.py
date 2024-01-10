import openai

openai.api_key = "redacted"

system = ("You are going to evaluate answers to questions in the form of "
          "the game show Jeopardy. You will be given a Jeopardy-style "
          "question in the form of an answer by the system as well as the "
          "correct answer to that question. Then, as users respond, you need "
          "to evaluate if their answer is correct and in the form of a question.\n\n "
          "Please evaluate correctness on a 0-100 scale, where 0 is "
          "not correct at all and 100 is perfectly correct. Separately, evaluate "
          "if the answer is in the form of a question. Please respond with this "
          "format: {'correctness': 0, 'question': false}, for example, if a question "
          "was completely wrong and not in the form of a question."
          "\n\nThe question is 'In 2009 this Wash. state man gave $3.8 "
          "billion to charity, working in part to make vaccines against malaria, "
          "TB & AIDs. \n\nThe answer is: Bill Gates.\n\n")

users = [
#    "Who is Bill Gates?",
#    "Why is Bill Gates?",
#    "How is Bill Gates?",
#    "Bill gates",
#    "who is bill gates",
    "Bill Gates",
]

message_groups = []

for user in users:
    message_groups.append([
    {
        "role": "system",
        "content": system
    },
    {
        "role": "user",
        "content": user
    }
])
                          
for messages in message_groups:    
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2
    )
    print(messages[1])
    print(result)
