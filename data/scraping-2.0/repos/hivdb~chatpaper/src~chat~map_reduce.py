from src.apis.chat_api import chat_ai
from src.apis.embedding import get_token_length
from .filter_question import get_unanswered
from src.logs import logger
from src.checksum import get_md5
from openai.error import Timeout
import warnings


def try_map_reduce(questions, chat_context):
    # TODO dry run mode
    segment_content(questions, chat_context)

    resp_list = []
    # TODO, add batch number in report?
    for i in chat_context['batches']:
        resp_list.append(
            process_one_batch(questions, i, chat_context))

    return resp_list


def segment_content(questions, chat_context):
    # TODO, save prompt

    prompts = []

    if chat_context['cheatsheet?']:
        prompts.append(chat_context['cheatsheet'])

    prompts.append(
        chat_context['question_template'].format(
            question=get_question_list_str(questions))
    )

    doc_parts = chat_context['doc_parts']

    [
        i.update({'token_length': get_token_length(i['all_content'])})
        for i in doc_parts
    ]

    result = []

    one_batch = []

    for i in doc_parts:
        new_content = i['all_content']

        parts = '\n'.join(one_batch + [new_content])

        p_prompt = chat_context['paper_template'].format(
            paper_content=parts)

        prompt = '\n'.join(prompts + [p_prompt])

        prompt_length = get_token_length(prompt, chat_context['model'])

        if prompt_length < chat_context['req_token_limit']:
            one_batch.append(new_content)
        else:
            one_batch_content = '\n'.join(one_batch)
            result.append({
                'batch_content': one_batch_content,
                'content_length': get_token_length(one_batch_content),
                'num_parts': len(one_batch),
            })
            one_batch = [new_content]

    if one_batch:
        one_batch_content = '\n'.join(one_batch)
        result.append({
            'batch_content': one_batch_content,
            'content_length': get_token_length(one_batch_content),
            'num_parts': len(one_batch),
        })

    chat_context['batches'] = result
    return result


def process_one_batch(questions, part, chat_context):

    if len(questions) > 1:
        try:
            process_multi_questions(questions, part, chat_context)
        except Timeout:
            for qid, q in questions.items():
                process_one_question({qid: q}, part, chat_context)
    else:
        process_one_question(questions, part, chat_context)


def process_one_question(unanswered, part, chat_context):

    unanswered = get_unanswered(
        unanswered, part['batch_content'],
        chat_context['chat_history'],
        chat_context['run_number'])

    if not unanswered:
        return

    unanswered = get_question_prompt(unanswered, chat_context)

    # print(chat_context['cheatsheet'])

    prompts = []

    if chat_context['cheatsheet?']:
        prompts.append(chat_context['cheatsheet'])

    prompts.append(
        chat_context['question_template'].format(
            question=get_question_list_str(unanswered))
    )

    prompts.append(
        chat_context['paper_template'].format(
            paper_content=part['batch_content']
        )
    )
    prompt = '\n'.join(prompts)
    logger.debug(prompt)

    resp = chat_ai(prompt, chat_context['model'])

    if resp['answer'] is None:
        resp['answer'] = 'AI answers NA'

    resp['question'] = list(unanswered.values())[0]
    resp['question_id'] = list(unanswered.keys())[0]

    # TODO, md5 check should not be in chat history, because
    # chat history is only for store the final answers, not part of answers
    # that need to be reduced.
    resp['md5'] = get_md5(part['batch_content'])
    resp['#batches'] = len(chat_context['batches'])
    resp['run_number'] = chat_context['run_number']

    chat_context['chat_history'].log(
        resp,
        {resp['question_id']: resp['question']})


def process_multi_questions(unanswered, part, chat_context, retry_time=10):
    # TODO, define retry 3 times

    unanswered = get_unanswered(
        unanswered, part['batch_content'],
        chat_context['chat_history'],
        chat_context['run_number'])

    unanswered = get_question_prompt(unanswered, chat_context)
    # print(chat_context['cheatsheet'])

    while len(unanswered) > 0 and retry_time:
        process_questions(part, unanswered, chat_context)

        unanswered = get_unanswered(
            unanswered, part['batch_content'],
            chat_context['chat_history'],
            chat_context['run_number'])

        # print('unanswered', unanswered)
        retry_time -= 1
        logger.info(f'Retry {retry_time}, remaining #{len(unanswered)}')


def get_question_list_str(questions):

    return "\n".join([
        f"{k}. {v}"
        for k, v in questions.items()
    ])


def process_questions(part, questions, chat_context):

    prompts = []

    if chat_context['cheatsheet?']:
        prompts.append(chat_context['cheatsheet'])

    question_str = get_question_list_str(questions)

    prompts.append(
        chat_context['question_template'].format(
            question=question_str)
    )

    prompts.append(
        chat_context['paper_template'].format(
            paper_content=part['batch_content'])
    )

    prompt = '\n'.join(prompts)
    logger.debug(prompt)

    # prompt_tokens = get_token_length(prompt)

    resp = chat_ai(prompt, chat_context['model'])
    if resp['answer'] is None:
        raise KeyError('content')
    resp['md5'] = get_md5(part['batch_content'])
    resp['#batches'] = len(chat_context['batches'])
    resp['run_number'] = chat_context['run_number']

    chat_context['chat_history'].log(resp, questions)

    return resp


def get_question_prompt(questions, chat_context):

    if chat_context['remove_sent?']:
        return remove_instruction(questions, chat_context)
    elif chat_context['append_sent?']:
        return append_instruction(questions, chat_context)

    return questions


def remove_instruction(questions, chat_context):

    cheatsheet = chat_context['cheatsheet']

    new_questions = {}
    for qid, q in questions.items():
        if '\n\n\n' not in q:
            warnings.warn(f"{qid}. {q} doesn't contain instruction")
            instr, q = '', q
        else:
            instr, q = q.split('\n\n\n')

        for j in instr.split('\n'):
            j = j.strip()
            if j not in cheatsheet:
                warnings.warn(f"{qid}. {q} instruction not found. {j}")
            cheatsheet = cheatsheet.replace(j, '')

        new_questions[qid] = q

    chat_context['cheatsheet'] = cheatsheet

    return new_questions


def append_instruction(questions, chat_context):

    cheatsheet = chat_context['cheatsheet']

    cheatsheet += '\n\n## Additional information\n\n'

    new_questions = {}
    for qid, q in questions.items():
        if '\n\n\n' not in q:
            raise Exception(f"{qid}. {q} doesn't contain instruction")
        instr, q = q.split('\n\n\n')

        cheatsheet += f'\n{instr}\n'

        new_questions[qid] = q

    chat_context['cheatsheet'] = cheatsheet

    return new_questions
