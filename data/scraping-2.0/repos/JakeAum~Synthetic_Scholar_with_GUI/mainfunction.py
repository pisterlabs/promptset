import openai
import re
import time

API_KEY = "YOUR_OPEN_AI_KEY"
model_engine = "YOUR_MODEL_ENGINE"

# Write to Classes
text = "YOUR_TEXT_HERE"
lines = text.splitlines()
with open('Inputs/course_list.txt', 'w') as file:
    for line in lines:
        line = line.strip()
        if line:
            file.write(line + '\n')

# Topic Generator
subject = "YOUR_SUBJECT_HERE"
openai.api_key = API_KEY
prompt = f"Create a list of class 50 unique topics (1-3 words long) commonly found in a college textbook table of contents for {subject} course in a bulleted format. Do not duplicate any topics and make sure they are not similar to each other."
response = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    temperature=0,
    max_tokens=1200,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
generated_text = response["choices"][0]["text"]
topic_pattern = re.compile(("(.)\s?(.+)"))
topics_found = topic_pattern.finditer(generated_text)
topics = []
for topic in topics_found:
    topics.append(topic.group(2).strip())

# New Prompts
openai.api_key = API_KEY
temperature = 1
subjects = open('Inputs/course_list.txt', 'r')
current_subject = ""
list_of_topics = []
current_topic = ""
for line in subjects.readlines():
    time.sleep(5)
    current_subject = repr(line).replace("\\n", "").replace("'", "")
    generated_topics = topics
    list_of_topics.append(generated_topics)

# Main GPT
openai.api_key = API_KEY
subject_pattern = re.compile("(.+):")
topic_pattern = re.compile("(\"|')([\w\s']+),?('|\")")
for line in subjects.readlines():
    list_of_topics = []
    subjects = subject_pattern.finditer(line)
    for subject in subjects:
        current_subject = subject.group(1)
    topics = topic_pattern.finditer(line)
    for topic in topics:
        list_of_topics.append(topic.group(2))
    for topic in list_of_topics:
        for prompt_index in range(4):
            prompts = [
                f"Act as if you are a student studying for your final exams. Write very detailed lecture notes on {topic} for the course {current_subject}. Please include relevant key concepts, definitions, rules, and examples within the notes. Be descriptive and thorough in your notes. Randomize the formatting.",
                f"Imagine you are a student studying for your final exams. Write a long and detailed list of sample exam problems on {topic} for the course {current_subject} with solutions included. Randomize the formatting.",
                f"In the format of a homework sheet with the title 'Extra Practice Problems', Create a long list of problems on {topic} for the course {current_subject} and for each question first explain how you would go about solving the problem, then solve the problem showing your work. Be sure to solve many problems incorrectly, and next to the solution denote whether it is correct or incorrect by writing '[CORRECT]' or '[INCORRECT]'. If it is incorrect, show the correct answer and explain how to get the correct answer. Randomize the formatting.",
                f"Imagine you are a
