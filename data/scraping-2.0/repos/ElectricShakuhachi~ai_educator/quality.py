from random import choice
import csv
import openai

test_count = 2

# AI TWEAKING SETTINGS FOR QUALITY CHECKER AI
q_model = "gpt-3.5-turbo"
q_temperature = 0.9
q_max_tokens = 2000
q_top_p = 1
q_frequency_penalty = 0.0
q_presence_penalty = 0.6
# END OF AI TWEAKING SETTINGS

# AI TWEAKING SETTINGS FOR SUBJECT CREATOR AI
s_model = "gpt-3.5-turbo"
s_temperature = 0.9
s_max_tokens = 2000
s_top_p = 1
s_frequency_penalty = 0.0
s_presence_penalty = 0.6
# END OF AI TWEAKING SETTINGS

def get_subject_prompt(grade, goal):
    return f"""Answer just a subject that might interest a student of grade {grade} - and for which there is enough source material so that it
    can be used to teach the goal: {goal}. Do not add generate anything else in your answer, but just the subject."""

def get_system_prompt(goal, grade, subject):
    prompt = " "
    return prompt

def generate_subject(prompt):
    res = openai.ChatCompletion.create(
        engine=s_model,
        prompt=prompt,
        temperature=s_temperature,
        max_tokens=s_max_tokens,
        top_p=s_top_p,
        frequency_penalty=s_frequency_penalty,
        presence_penalty=s_presence_penalty
    )
    return res.choices[0]["message"]["content"]

if __name__ == "__main__":
    for i in range(test_count):
        student_grade = choice(range(1, 13))
        with open("ccss.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        goal = choice(reader)[0]
        subject = generate_subject(get_subject_prompt(student_grade, goal)
        system_prompt = get_system_prompt(goal, student_grade, subject)
        res = openai.ChatCompletion.create(
            engine=q_model,
            prompt=system_prompt,
            temperature=q_temperature,
            max_tokens=q_max_tokens,
            top_p=q_top_p,
            frequency_penalty=q_frequency_penalty,
            presence_penalty=q_presence_penalty
        )
