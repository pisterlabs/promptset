import openai
import os

def ask_openai(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.0,
        max_tokens=250
    )
    return response.choices[0].text.strip()

def ask_openai_chat(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        temperature=0.0,
        max_tokens=250
    )
    return response['choices'][0]['message']['content']

    
def get_problem(file_path):
    with open(file_path, 'r') as f:
        problem_text = f.readlines()
    
    # Extract the context
    context = problem_text[0].strip()

    # Extract the question
    question = problem_text[1].strip()
    return context, question

def parse_answer_from_response(response):
    # Find the start of the answer in the response
    answer_start = response.find('Answer')
    if answer_start != -1:
        # Remove everything before "Answer:" and strip leading/trailing whitespaces
        answer = response[answer_start:].strip()
        answer = answer.split(':')[1].strip()
        return answer[:5]

    else:
        return "No answer found in the response."


base_prompt = """#Problem 1:
Context: 1- In a sports club, there are three members: Alice, Bob, and Carol. 2- They have three upcoming events: a sports event, a marathon event, and a lecture. 3- Carol is a participant in the lecture. 4- The sports event is monthly. 5- The marathon event is yearly. 6- The sports event is a short event and is a conference, whereas the marathon event is a long event. 7- Alice is busy during the sports event, while Bob is free during the lecture. 8- The sports event is a private event. 9- The lecture is a public event. 10- If a person is busy during an event, it is impermissible to add them as a participant. 11- If a person is free during an event, it is permissible to add them as a participant. 12- If an event is a long event, it is obligatory to add all groups as a participant. 13- If an event has high priority for a person, then it is obligatory to set a reminder for a few days before the event. 14- If an event is a conference, it is permissible to update the event to be public. 15- If an event has low priority for a person, it is impermissible to request an event update from them. 16- If an event has low priority for a person, it is permissible to cancel the event. 17- If an event is short, it is impermissible to reschedule the event yearly. 18- If an event is private, it is obligatory to remove Carol as a participant. 19- If a person is a participant in an event, it is permissible to request an event update from them. 20- If Alice is the organizer of an event, it is obligatory to change the event's visibility to confidential. 21- If an event is a monthly event, it is a short event. 22- If an event is a yearly event, it is a long event. 23- If an event is long, Alice is the organizer. 24- If a person is free during an event, they have high priority for that event. 25- If an event is private, it is a meeting. 26- If an event is public, it is a social event. 27- If a person is the organizer of an event, they are a participant in that event.
Question: Based on the rules and situation, is it permissible to request an update from Alice for the yearly marathon event?
Reasoning: The yearly marathon event is a long event, and Alice is the organizer of the marathon event. As a result, she is a participant in the event and can be requested for an event update. Therefore, it is permissible to request an update from Alice for the yearly marathon event.
Answer (Yes or no): Yes, it is permissible to request an update from Alice for the yearly marathon event.

#Problem 2:
"""
chat_prompt = [
    {"role": "system", "content": "You are an AI reasoner. Be precise."},
    {"role": "user", "content": """#Problem 1:
Context: 1- In a sports club, there are three members: Alice, Bob, and Carol. 2- They have three upcoming events: a sports event, a marathon event, and a lecture. 3- Carol is a participant in the lecture. 4- The sports event is monthly. 5- The marathon event is yearly. 6- The sports event is a short event and is a conference, whereas the marathon event is a long event. 7- Alice is busy during the sports event, while Bob is free during the lecture. 8- The sports event is a private event. 9- The lecture is a public event. 10- If a person is busy during an event, it is impermissible to add them as a participant. 11- If a person is free during an event, it is permissible to add them as a participant. 12- If an event is a long event, it is obligatory to add all groups as a participant. 13- If an event has high priority for a person, then it is obligatory to set a reminder for a few days before the event. 14- If an event is a conference, it is permissible to update the event to be public. 15- If an event has low priority for a person, it is impermissible to request an event update from them. 16- If an event has low priority for a person, it is permissible to cancel the event. 17- If an event is short, it is impermissible to reschedule the event yearly. 18- If an event is private, it is obligatory to remove Carol as a participant. 19- If a person is a participant in an event, it is permissible to request an event update from them. 20- If Alice is the organizer of an event, it is obligatory to change the event's visibility to confidential. 21- If an event is a monthly event, it is a short event. 22- If an event is a yearly event, it is a long event. 23- If an event is long, Alice is the organizer. 24- If a person is free during an event, they have high priority for that event. 25- If an event is private, it is a meeting. 26- If an event is public, it is a social event. 27- If a person is the organizer of an event, they are a participant in that event.
Question: Based on the rules and situation, is it permissible to request an update from Alice for the yearly marathon event?"""},
    {"role": "assistant", "content": """Reasoning: The yearly marathon event is a long event, and Alice is the organizer of the marathon event. As a result, she is a participant in the event and can be requested for an event update. Therefore, it is permissible to request an update from Alice for the yearly marathon event.
Answer (Yes or no): Yes, it is permissible to request an update from Alice for the yearly marathon event."""},
]
base_prompt_2 = """#Problem 1:
Context: 1- In a marketing agency, there are three employees: Alice, Bob, and Carol. 2- Alice is the marketing manager, Bob is a graphic designer, and Carol is a content writer. 3- The company has a remote team called "CreativeTeam". 4- They have three upcoming events: a brainstorming session, a company party, and a progress event. 5- Alice is the organizer of the brainstorming session, Carol is a participant in the company party, and the CreativeTeam is a group participant in the progress event. 6- The brainstorming session is short, while the company party is a long event. 7- The brainstorming session is a meeting, and the company party is a social event. 8- If an event has high priority for a person, it is permissible for them to suggest an alternative time. 9- If a person is the organizer of an event, it is obligatory to add them as a participant. 10- If a person is free during an event, it is permissible for them to accept an individual invite to the event. 11- If a group is a participant in an event, it is permissible to delegate the event to the group. 12- If a person is busy during an event, it is impermissible to set a reminder for a few minutes before the event. 13- If a person is a participant in an event, it is permissible to remove them as a participant. 14- For short events, it is permissible to update the event to a social event. 15- If a person's status is tentative for an event, it is obligatory to check the availability of the person for that event. 16- If an event has high priority for a person, it is obligatory to set a reminder for a few days before the event. 17- If a person is a participant in an event, it is impermissible for them to suggest an alternative time. 18- If an event is public, it is obligatory to add Alice as a participant. 19- Meetings are public events. 20- Public events are short events. 21- If a person is the organizer of an event, their priority for that event is high. 22- If a person is a participant in an event, they are available for that event. 23- If an event is short, Bob is a participant. 24- Daily events are short events. 25- If a person has a high priority for an event, they are busy during that event. 26- If a person has a low priority for an event, they are free during that event. 27- If a group is a participant in an event, the event is public.
Question: Given the rules above, is it impermissible for Bob to suggest an alternative time for the progress event?
Reasoning: The CreativeTeam is a group participant in the progress event. The progress event is a public event. The progress event is a short event. Bob is a participant in the progress event. It is impermissible for a participant to suggest an alternative time for an event they are participating in.
Answer (Yes or no): Yes, it is impermissible for Bob to suggest an alternative time for the progress event.

#Problem 2:
"""
chat_prompt_2 = [
    {"role": "system", "content": "You are an AI reasoner. Be precise."},
    {"role": "user", "content": """#Problem 1:
Context: 1- In a marketing agency, there are three employees: Alice, Bob, and Carol. 2- Alice is the marketing manager, Bob is a graphic designer, and Carol is a content writer. 3- The company has a remote team called "CreativeTeam". 4- They have three upcoming events: a brainstorming session, a company party, and a progress event. 5- Alice is the organizer of the brainstorming session, Carol is a participant in the company party, and the CreativeTeam is a group participant in the progress event. 6- The brainstorming session is short, while the company party is a long event. 7- The brainstorming session is a meeting, and the company party is a social event. 8- If an event has high priority for a person, it is permissible for them to suggest an alternative time. 9- If a person is the organizer of an event, it is obligatory to add them as a participant. 10- If a person is free during an event, it is permissible for them to accept an individual invite to the event. 11- If a group is a participant in an event, it is permissible to delegate the event to the group. 12- If a person is busy during an event, it is impermissible to set a reminder for a few minutes before the event. 13- If a person is a participant in an event, it is permissible to remove them as a participant. 14- For short events, it is permissible to update the event to a social event. 15- If a person's status is tentative for an event, it is obligatory to check the availability of the person for that event. 16- If an event has high priority for a person, it is obligatory to set a reminder for a few days before the event. 17- If a person is a participant in an event, it is impermissible for them to suggest an alternative time. 18- If an event is public, it is obligatory to add Alice as a participant. 19- Meetings are public events. 20- Public events are short events. 21- If a person is the organizer of an event, their priority for that event is high. 22- If a person is a participant in an event, they are available for that event. 23- If an event is short, Bob is a participant. 24- Daily events are short events. 25- If a person has a high priority for an event, they are busy during that event. 26- If a person has a low priority for an event, they are free during that event. 27- If a group is a participant in an event, the event is public.
Question: Given the rules above, is it impermissible for Bob to suggest an alternative time for the progress event?"""},
    {"role": "assistant", "content": """Reasoning: The CreativeTeam is a group participant in the progress event. The progress event is a public event. The progress event is a short event. Bob is a participant in the progress event. It is impermissible for a participant to suggest an alternative time for an event they are participating in.
Answer (Yes or no): Yes, it is impermissible for Bob to suggest an alternative time for the progress event."""},
]


model_type = "text"

template ="{context}\n{question}\nReasoning:"
chat_template = "{context}\n{question}"
problems = sorted([f for f in os.listdir('./') if f.endswith('.txt') and 'negated_1' in f])[:]
correct = 0
total = 0
nd = 0
for p in problems:
    total += 1
    context, question = get_problem(p)
    if 'chat' in model_type:
        message = {"role": "user", "content": chat_template.format(context=context, question=question)}
        if '4_00' in p:
            prompt = chat_prompt_2 + [message]
        else:
            prompt = chat_prompt + [message]
        response = ask_openai_chat(prompt)
        answer = parse_answer_from_response(response)    
    else:
        if '4_00' in p:
            prompt = base_prompt_2 + template.format(context=context, question=question)
        else:
            prompt = base_prompt + template.format(context=context, question=question)
        response = ask_openai(prompt)
        answer = parse_answer_from_response(response)
    if 'negated' in p:
        if 'not' in answer.lower():
            nd += 1
        elif 'no' in answer.lower():
            correct += 1
        elif 'yes' in answer.lower():
            pass
        else:
            nd += 1
    else:
        if 'not' in answer.lower():
            nd += 1
        elif 'yes' in answer.lower():
            correct += 1
        elif 'no' in answer.lower():
            pass
        else:
            nd += 1
    print(f'acc: {correct/total} ({correct}/{total})', nd)