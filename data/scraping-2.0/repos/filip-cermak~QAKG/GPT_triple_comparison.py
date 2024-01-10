import random
import re
import openai

openai.api_key = "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"

def make_query(q):
    s = "Given the semantic triples, find which group of triples is the most consistent with the triples from the context, either A, B, C or D. Return answer only: "

    choices = ["A", "B", "C", "D"]

    options = {}
    options[0] = [[t.subject, t.relation, t.object] for t in q.question_with_distractors_triples[0]]
    options[1] = [[t.subject, t.relation, t.object] for t in q.question_with_distractors_triples[1]]
    options[2] = [[t.subject, t.relation, t.object] for t in q.question_with_distractors_triples[2]]
    options[100] = [[t.subject, t.relation, t.object] for t in q.question_with_answer_triples]

    options_list = list(options.items())

    random.shuffle(options_list)
    shuffle_order = [i[0] for i in options_list]

    correct = choices[shuffle_order.index(100)]

    out = {}    
    out['context'] = [[t.subject, t.relation, t.object] for t in q.context_triples]

    tmp = {choices[i]:l[1] for i, l in enumerate(options_list)}

    out = out | tmp

    query = s+str(out)
    query = query.replace('"', "")
    query = query.replace("\'", "")

    return query, correct

def extract_answer(s):
    pattern = r'\b(A|B|C|D)\b'
    match = re.search(pattern, s)

    if match:
        return match.group()
    else:
        return None

def get_chat_gpt_response(q):

    query, correct = make_query(q)

    MODEL = "gpt-3.5-turbo"
    
    response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": query},
    ],
    temperature=0,
    )

    return extract_answer(response['choices'][0]['message']['content']), correct, query, response
