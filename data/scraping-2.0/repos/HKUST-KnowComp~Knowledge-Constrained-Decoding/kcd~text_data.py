from functools import partial

from kcd.instructions import OPENAI_INSTRUCTION, get_instruction
from kcd.kilt.load_kilt_data import load_fever
from kcd.dstc11_task5 import load_dstc_data
from kcd.summarization import load_summary_data


def load_text_data(path='data/fever-dev-kilt-processed.jsonl',
                   use_kilt_format=True,
                   instruction_model='openai',
                   task='chat',
                   tokenize=False,
                   tokenizer=None,
                   add_trailing_newline=False,
                   no_label=False):
    if 'fever' in path:
        load_fn = load_fever
        data_task = 'fever'
    elif 'wow' in path:
        if use_kilt_format:
            from kcd.kilt.load_kilt_data import load_wow
        else:
            from kcd.wizard_of_wikipedia import load_wow
        load_fn = load_wow
        data_task = 'wow'
    elif 'dstc11_task5' in path:
        load_fn = load_dstc_data
        data_task = 'dstc11_task5'
    elif 'cnn_dailymail' in path or 'xsum' in path:
        load_fn = load_summary_data
        data_task = 'summarization'
    else:
        raise ValueError(f'Unknown dataset: {path}')

    def prepare(config, example):
        # ['question', 'answers', 'id', 'ctxs', 'label']
        knowledge = example['ctxs'].strip()
        question = example['question'].strip()
        answer = example['answers'].strip()
        if task != 'chat':
            if 'question' in config['input_columns']:
                input_text = get_instruction(instruction_model,
                                             data_task,
                                             question=question,
                                             knowledge=knowledge)
                target = answer
            else:
                input_text = get_instruction(instruction_model, data_task, knowledge=knowledge)
                target = question

            if tokenize:
                assert tokenizer is not None
                if no_label:
                    target = None
                if add_trailing_newline:
                    input_text = input_text + '\n\n'
                tokenized = tokenizer(input_text,
                                      text_target=target,
                                      return_tensors='pt',
                                      truncation=True,
                                      max_length=tokenizer.model_max_length)
                return {k: v[0] for k, v in tokenized.items()}
        else:
            if data_task == 'summarization':
                # summarization
                template = "Summarize the following text:\n\n{}"
                input_text = [{'role': 'user', 'content': template.format(knowledge)}]
                target = answer
                return {'prompt': input_text, 'completion': target}
            # wow for openai chatCompletion API
            if '</eot>' in question:  # dstc11_task5
                conversation_history = question.split('</eot>')
                conversation_history = [
                    txt.replace('User: ', '').replace('System: ', '')
                    for txt in conversation_history
                ]
            else:
                conversation_history = question.split('\n')
            # 1: wizard = assistant, 0: user
            user = list(map(int, example['user'].split(',')))
            if len(conversation_history) > len(user):
                assert len(conversation_history) == len(user) + 1
                first_utt = '\n'.join(conversation_history[:2])
                conversation_history = [first_utt] + conversation_history[2:]

            messages = [{
                "role": "user" if i == 0 else "assistant",
                "content": chat
            } for i, chat in zip(user, conversation_history)]
            messages.append({"role": "system", "content": OPENAI_INSTRUCTION.format(knowledge)})

            input_text = messages
            target = answer

        return {'prompt': input_text, 'completion': target}

    dataset, config = load_fn(path)
    filtered = dataset.filter(lambda x: x['ctxs'] is not None)
    mapped = filtered.map(partial(prepare, config), remove_columns=dataset.column_names)
    return mapped
