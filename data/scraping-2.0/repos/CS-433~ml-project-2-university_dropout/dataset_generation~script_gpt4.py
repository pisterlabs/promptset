import requests
import json
import re
import os
import sys

from llama_index.llms import OpenAI

path_to_data = sys.argv[1]
store_path = sys.argv[2]

with open(path_to_data, 'r') as f:
    data = json.load(f)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def generate_qa(transcript):
    inp = ("There is an excerpt in French from RTS (Radio Télévision Suisse) below:\n"
            f"{transcript}"
            "\n\nGenerate three questions in French based solely on this excerpt (this excerpt must contian the right answer for each of them) "
            "and generate right answers for each of them in French based solely on this text. You must not use direct references to the "
            "passage provided (such as \"dans le texte\", \"dans le extrait\", etc.) as the interlocutor does not have access to this passage. "
            "Answers must be detailed and complete. "
            "\n\nThe format of your response:\n\n"
            "Question : Exemple de question 1 ?\n"
            "Réponse : Exemple de réponse 1\n"
            "Question : Exemple de question 2 ?\n"
            "Réponse : Exemple de réponse 2\n"
            "Question : Exemple de question 3 ?\n"
            "Réponse : Exemple de réponse 3")
    
    result = client.complete(inp, model="gpt-4-1106-preview").text
    qas = result.split('Question : ')[1:]

    questions, answers = [], []

    for qa in qas:
        try:
            qa_split = re.split('Réponse : ', qa)
            question, answer = qa_split
        except:
            continue
        question, answer = question.strip(), answer.strip()
        if not question[-1] == '?' or len(question) > 200:
            continue
        print(question)
        print(answer)
        questions.append(question.strip())
        answers.append(answer.strip())
    
    return questions, answers

result = []
for i, example in enumerate(data):
    if i % 10 == 4:
        with open(store_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    transcript = example['transcript']
    if len(transcript) < 1000 or len(transcript) > 10000:
        continue

    media_id = example['media_id']
    questions, answers = generate_qa(transcript)
    if questions == -1:
        continue

    for question, answer in zip(questions, answers):
        result.append({
            'question': question,
            'answer': answer,
            'transcript': transcript,
            'media_id': media_id
        })

    if len(result) >= 100:
        break
    print(len(result))

with open(store_path, 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)