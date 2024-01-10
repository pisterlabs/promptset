import json
import openai

openai.api_key = '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'
openai.api_base = "https://closeai.deno.dev/v1"
task_intro = 'You will be given one summary written for 8 reviews. ' \
             'Your task is to rate the summary on one metric. ' \
             'Please make sure you read and understand these instructions carefully. ' \
             'Please keep this document open while reviewing, and refer to it as needed.'
asp_rel_eval_crit = 'Evaluation Criteria:\n' \
                    'Aspect Relevance (1-5) - measures whether the mainly discussed aspects in the reviews ' \
                    'are covered exactly by the summary.'
self_coh_eval_crit = 'Evaluation Criteria:\n' \
                     'Self-Coherence (1-5) - measures whether the summary is consistent within itself ' \
                     'in terms of sentiments and aspects.'
sent_con_eval_crit = 'Evaluation Criteria:\n' \
                     'Sentiment Consistency (1-5) - measures whether the summary is consistent with the reviews ' \
                     'in terms of sentiments for each aspect.'
readability_eval_crit = 'Evaluation Criteria:\n' \
                        'Readability (1-5) - measures whether the summary is fluent and informative.'
dimension = ['Aspect Relevance', 'Self-Coherence', 'Sentiment Consistency', 'Readability']
crits = [asp_rel_eval_crit, self_coh_eval_crit, sent_con_eval_crit, readability_eval_crit]

dim_instr_dict = {}
for dim, eval_crit in zip(dimension, crits):
    prompt = task_intro + '\n' + eval_crit + '\n' + 'Evaluation Steps:\n'

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=2048,
        temperature=0,
        echo=True
    )
    print('completed')
    text = response["choices"][0]['text']
    dim_instr_dict[dim] = text

f = open('geval_instructions_origin.jsonl', 'w')
f.write(json.dumps(dim_instr_dict)+'\n')
f.close()
