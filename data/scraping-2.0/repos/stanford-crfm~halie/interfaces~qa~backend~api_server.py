import os
import gc
import shutil
import random
import openai
import warnings
import numpy as np
from time import time
from argparse import ArgumentParser

from api_key import (
    read_api_keys, setup_crfm,
    set_default_api_keys,
    set_openai_api_key_for_domain, set_crfm_api_key_for_domain,
)
from reader import (
    read_log,
    read_examples, read_prompts, read_blocklist,
    read_access_codes, read_access_code_history,
    update_session_id_history,
    read_iqa_questions, read_iqa_sequences,
    read_summarization_samples, read_summarization_sequences,
)
from helper import (
    print_verbose, print_current_sessions,
    get_uuid, retrieve_log_paths,
    append_session_to_file,
    save_log_to_jsonl, compute_stats, get_last_text_from_log, get_config_for_log,
    is_expired,  convert_iqa_sequence_to_data,
    convert_summarization_sequence_to_data,
)
from parsing import (
    parse_prompt, parse_suggestion, parse_probability,
    filter_suggestions
)

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

warnings.filterwarnings("ignore", category=FutureWarning)  # noqa

SESSIONS = dict()
app = Flask(__name__)
CORS(app)  # For Access-Control-Allow-Origin

SUCCESS = True
FAILURE = False


@app.route('/api/start_session', methods=['POST'])
@cross_origin(origin='*')
def start_session():
    content = request.json
    result = {}

    # Read newest prompts, examples, and access codes (removed to conserve memory)
    global examples, prompts
    examples = read_examples(args.example_path)
    prompts = read_prompts(args.prompt_path)
    allowed_access_codes = read_access_codes(args.data_dir, len(iqa_sequences))

    # Check access code
    access_code = content['accessCode']

    if access_code not in allowed_access_codes:
        if not access_code:
            access_code = '(not provided)'
        result['status'] = FAILURE
        result['message'] = f'Invalid access code: {access_code}. Please check your access code in URL.'
        print_current_sessions(SESSIONS, 'Invalid access code')
        return jsonify(result)

    config = allowed_access_codes[access_code]

    # Check if access code is expired (removed to conserve memory)
    # start_date = config.start_date
    # end_date = config.end_date
    # expire_after = config.expire_after
    # access_code_history = read_access_code_history(args.session_id_history_path)
    # if is_expired(access_code, start_date, end_date, expire_after, access_code_history):
    #     result['status'] = FAILURE
    #     result['message'] = f'Expired access code: {access_code}.'
    #     print_current_sessions(SESSIONS, 'Expired access code')
    #     return jsonify(result)

    # Check if permission is granted for reading and saving files
    # as well as if there is enough disk space left for saving logs
    # Randomly check it with 2% of chance
    if args.check_disk_storage or random.randint(0, 50) == 50:
        try:
            if os.path.exists(args.log_big_size_path):
                src = args.log_big_size_path
                dst = os.path.join(args.log_dir, 'log_big_size_sample.json')

                if src == dst:
                    raise RuntimeError('Need to set log_dir that is different from all_log_dir')

                shutil.copy(src, dst)
                os.remove(dst)
        except Exception as e:
            result['status'] = FAILURE
            result['message'] = f'There was an error while starting your session. Please refresh this page and try again!'
            print_current_sessions(SESSIONS, f'Failed to save a log file with big size: {e}')
            return jsonify(result)

    # Setup a new session for user
    # If it's reedsy domain, add prefix `reedsy` to session ID
    # try:
    session_id = get_uuid()  # Generate unique ID

    if content['domain'] == 'reedsy':
        session_id = 'reedsy' + session_id

    verification_code = session_id

    config.assign_random_values()

    # Information returned to user
    result = {
        'access_code': access_code,
        'session_id': session_id,

        'example_text': examples[config.example],
        'prompt_text': prompts[config.prompt],
    }
    result.update(config.convert_to_dict())

    # Add questions and answers for interactive QA
    if int(config.iqa_sequence_index) > -1:
        interactiveqa_data = convert_iqa_sequence_to_data(
            iqa_sequences[int(config.iqa_sequence_index)],
            iqa_questions,
            args.num_questions_per_sequence,
        )
        result.update({
            'data': interactiveqa_data,
        })

    # Add samples for text summarization
    if int(config.summarization_sequence_index) > -1:
        summarization_data = convert_summarization_sequence_to_data(
            summarization_sequences[int(config.summarization_sequence_index)],
            summarization_samples,
        )
        print('summarization_data:', summarization_data)
        result.update({
            'data': summarization_data,
        })

    # Information stored in the server
    SESSIONS[session_id] = {
        'access_code': access_code,
        'session_id': session_id,

        'start_timestamp': time(),
        'last_query_timestamp': time(),
        'verification_code': verification_code,
    }
    SESSIONS[session_id].update(config.convert_to_dict())

    result['status'] = SUCCESS

    session = SESSIONS[session_id]
    model_name = ' '.join([result['engine'], result['model'], result['crfm']]).strip()
    domain = result['domain'] if 'domain' in result else ''

    append_session_to_file(session, args.session_id_history_path)
    print_verbose('New session created', session, args.verbose)
    print_current_sessions(SESSIONS, f'Session {session_id} ({domain}: {model_name}) has been started successfully.')
    # except Exception as e:
    #     result['status'] = FAILURE
    #     result['message'] = str(e)
    #     print_current_sessions(SESSIONS, f'Failed to create a new session: {e}')

    gc.collect(generation=2)
    return jsonify(result)


@app.route('/api/end_session', methods=['POST'])
@cross_origin(origin='*')
def end_session():
    content = request.json
    session_id = content['sessionId']
    log = content['logs']

    path = os.path.join(args.log_dir, session_id) + '.jsonl'

    results = {}
    results['path'] = path
    try:
        save_log_to_jsonl(path, log)
        results['status'] = SUCCESS
    except Exception as e:
        results['status'] = FAILURE
        results['message'] = str(e)
        print(e)
    print_verbose('Save log to file', {
        'session_id': session_id,
        'len(log)': len(log),
        'status': results['status'],
    }, args.verbose)

    # Remove finished session
    try:
        # NOTE Somehow end_session is called twice;
        # Do not pop session_id from SESSIONS to prevent exception
        session = SESSIONS[session_id]
        results['verification_code'] = session['verification_code']
        print_current_sessions(SESSIONS, f'Session {session_id} has been saved successfully.')
    except Exception as e:
        print(e)
        print('Error at the end of end_session; ignore')
        results['verification_code'] = 'SERVER_ERROR'
        print_current_sessions(SESSIONS, f'Session {session_id} has not been saved.')

    gc.collect(generation=2)
    return jsonify(results)


@app.route('/api/query', methods=['POST'])
@cross_origin(origin='*')
def query():
    content = request.json

    session_id = content['session_id']
    domain = content['domain']
    prev_suggestions = content['suggestions']

    results = {}

    try:
        SESSIONS[session_id]['last_query_timestamp'] = time()
    except Exception as e:
        print(f'Ignoring an error in query: {e}')

    # Check if session_id is valid
    if session_id not in SESSIONS:
        results['status'] = FAILURE
        results['message'] = f'Your session has not been established due to invalid access code. Please check your access code in URL.'
        return jsonify(results)

    example = content['example']
    example_text = examples[example]

    # Overwrite example text if it is manually provided
    if 'example_text' in content:
        example_text = content['example_text']

    # Get configuration
    n = int(content['n'])
    max_tokens = int(content['max_tokens'])
    temperature = float(content['temperature'])
    top_p = float(content['top_p'])
    presence_penalty = float(content['presence_penalty'])
    frequency_penalty = float(content['frequency_penalty'])

    engine = content['engine'] if 'engine' in content else None
    model = content['model'] if 'model' in content else None
    crfm = content['crfm'] if 'crfm' in content else None
    crfm_base = content['crfm_base'] if 'crfm_base' in content else None
    crfm_desired = content['crfm_desired'] if 'crfm_desired' in content else None

    stop = [sequence for sequence in content['stop'] if len(sequence) > 0]
    if 'DO_NOT_STOP' in stop:
        stop = []

    # Remove special characters
    stop_sequence = [sequence for sequence in stop if sequence not in {'.'}]
    stop_rules = [sequence for sequence in stop if sequence in {'.'}]
    if not stop_sequence:
        stop_sequence = None

    # Parse doc
    doc = content['doc']
    results = parse_prompt(example_text + doc, max_tokens)
    prompt = results['effective_prompt']

    # Query GPT-3
    try:
        if crfm:  # Query through CRFM API
            from benchmarking.src.common.request import Request
            set_crfm_api_key_for_domain(api_keys, domain)

            if not stop_sequence:
                stop_sequence = []

            if crfm_desired and crfm_desired != 'na':
                crfm = crfm_desired

            crfm_request = Request(
                model=crfm,
                prompt=prompt,
                num_completions=n,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stop_sequences=stop_sequence,

                random=str(np.random.randint(0, 100000)),
            )
            crfm_response = service.make_request(auth, crfm_request)
            suggestions = []
            for completion in crfm_response.completions:
                suggestion = parse_suggestion(
                    completion.text,
                    results['after_prompt'],
                    stop_rules
                )
                probability = completion.logprob
                suggestions.append((suggestion, probability, crfm))
                print(f'{crfm}: {suggestion}')

            if crfm_base:
                crfm_request = Request(
                    model=crfm_base,
                    prompt=prompt,
                    num_completions=n,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    stop_sequences=stop_sequence,

                    random=str(np.random.randint(0, 100000)),
                )
                crfm_response = service.make_request(auth, crfm_request)

                for completion in crfm_response.completions:
                    suggestion = parse_suggestion(
                        completion.text,
                        results['after_prompt'],
                        stop_rules
                    )
                    probability = completion.logprob
                    suggestions.append((suggestion, probability, crfm_base))
                    print(f'{crfm_base}: {suggestion}')

        elif model:  # Fine-tuned GPT-3
            set_openai_api_key_for_domain(api_keys, domain)

            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logprobs=10,
                stop=stop_sequence,
            )
            suggestions = []
            for choice in response['choices']:
                suggestion = parse_suggestion(
                    choice.text,
                    results['after_prompt'],
                    stop_rules
                )
                probability = parse_probability(choice.logprobs)
                suggestions.append((suggestion, probability, model))

        elif engine:  # One of the default GPT-3 models
            set_openai_api_key_for_domain(api_keys, domain)

            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logprobs=10,
                stop=stop_sequence,
            )
            suggestions = []
            for choice in response['choices']:
                suggestion = parse_suggestion(
                    choice.text,
                    results['after_prompt'],
                    stop_rules
                )
                probability = parse_probability(choice.logprobs)
                suggestions.append((suggestion, probability, engine))

        else:
            raise RuntimeError('None of engine, model, and crfm is specified.')
    except Exception as e:
        results['status'] = FAILURE
        results['message'] = str(e)
        print(e)
        return jsonify(results)

    # Always return original model outputs
    original_suggestions = []
    for index, (suggestion, probability, source) in enumerate(suggestions):
        original_suggestions.append({
            'original': suggestion,
            'trimmed': suggestion.strip(),
            'probability': probability,
            'source': source,
        })

    # Filter out model outputs for safety
    filtered_suggestions, counts = filter_suggestions(
        suggestions,
        prev_suggestions,
        blocklist,
    )

    random.shuffle(filtered_suggestions)

    suggestions_with_probabilities = []
    for index, (suggestion, probability, source) in enumerate(filtered_suggestions):
        suggestions_with_probabilities.append({
            'index': index,
            'original': suggestion,
            'trimmed': suggestion.strip(),
            'probability': probability,
            'source': source,
        })

    results['status'] = SUCCESS
    results['original_suggestions'] = original_suggestions
    results['suggestions_with_probabilities'] = suggestions_with_probabilities
    results['ctrl'] = {
        'n': n,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'presence_penalty': presence_penalty,
        'frequency_penalty': frequency_penalty,
        'stop': stop,
    }
    results['counts'] = counts
    print_verbose('Result', results, args.verbose)
    return jsonify(results)


@app.route('/api/save_log', methods=['POST'])
@cross_origin(origin='*')
def save_log():
    content = request.json
    session_id = content['sessionId']
    logs = content['logs']

    results = dict()
    if not session_id.startswith('reedsy'):
        results['status'] = FAILURE
        results['message'] = 'You can only overwrite logs with session IDs starting with "reedsy."'
    else:
        results['status'] = SUCCESS
        path = os.path.join(args.log_dir, session_id) + '.jsonl'
        save_log_to_jsonl(path, logs)
    return jsonify(results)


@app.route('/api/get_log', methods=['POST'])
@cross_origin(origin='*')
def get_log():
    results = dict()

    content = request.json
    session_id = content['sessionId']
    domain = content['domain'] if 'domain' in content else None

    # Retrieve the latest list of logs
    log_paths = retrieve_log_paths(args.all_log_dir)

    try:
        log_path = log_paths[session_id]
        log = read_log(log_path)
        results['status'] = SUCCESS
        results['logs'] = log
    except Exception as e:
        results['status'] = FAILURE
        results['message'] = str(e)

    if results['status'] == FAILURE:
        return results

    # Populate metadata
    try:
        stats = compute_stats(log)
        last_text = get_last_text_from_log(log)
        config = get_config_for_log(
            session_id,
            session_id_history,
            args.session_id_history_path
        )
    except Exception as e:
        print(f'Failed to poopulate metadata for the log: {e}')
        stats = None
        last_text = None
        config = None
    results['stats'] = stats
    results['config'] = config
    results['last_text'] = last_text

    # Reestablish a session based on the metadata (only for reedsy)
    if config and domain == 'reedsy':
        try:
            SESSIONS[session_id] = config
            SESSIONS[session_id]['start_timestamp'] = time()
            SESSIONS[session_id]['last_query_timestamp'] = time()

            model_name = ' '.join([
                config['engine'], config['model'], config['crfm']
            ]).strip()
            domain = config['domain'] if 'domain' in config else ''
            print_current_sessions(
                SESSIONS,
                f'Session {session_id} ({domain}: {model_name}) has been loaded successfully.'
            )
        except Exception as e:
            print_current_sessions(SESSIONS, f'Failed to load an existing session: {e}')

    if args.verbose:
        print_verbose('Get log', results, args.verbose)
    return results


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--api_key_path', type=str)
    parser.add_argument('--crfm_dir', type=str)

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--session_id_history_path', type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--log_big_size_path', type=str)

    parser.add_argument('--example_path', type=str)
    parser.add_argument('--prompt_path', type=str)
    parser.add_argument('--blocklist_path', type=str)
    parser.add_argument('--use_blocklist', action='store_true')

    # For interactive QA
    parser.add_argument('--iqa_question_path', type=str)
    parser.add_argument('--iqa_sequence_path', type=str)
    parser.add_argument('--num_questions_per_sequence', type=int)

    # For text summarization
    parser.add_argument('--summarization_sample_path', type=str)
    parser.add_argument('--summarization_sequence_path', type=str)

    parser.add_argument('--port', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--check_disk_storage', action='store_true')

    parser.set_defaults(
        api_key_path='./credentials/api_key.csv',
        data_dir='../data',  # Read all csv files with "access_code" in file name
        session_id_history_path='../logs/session.txt',
        all_log_dir='../logs',
        log_dir='../logs',
        log_big_size_path='../logs/log_big_size_sample.json',

        example_path='../data/examples',
        prompt_path='../data/prompts.tsv',
        blocklist_path='../data/blocklist.txt',

        iqa_question_path='../data/iqa/all_questions_large.csv',
        iqa_sequence_path='../data/iqa/question_sequences_10.csv',
        num_questions_per_sequence=11,

        summarization_sample_path='../data/summarization/samples_summarization.csv',
        summarization_sequence_path='../data/summarization/sequences_summarization.txt',

        port=5555,
        debug=False,
        verbose=False,
    )

    global args
    args = parser.parse_args()

    # Read and set default API keys
    global api_keys
    api_keys = read_api_keys(args.api_key_path)

    try:
        global auth, service
        auth, service = setup_crfm(args.crfm_dir, api_keys)
    except Exception as e:
        print(f'Could not set CRFM API properly: {e}')

    set_default_api_keys(api_keys)

    # Create session txt if necessary
    if not os.path.exists(args.session_id_history_path):
        with open(args.session_id_history_path, 'w') as f:
            f.write('')

    # Read examples, prompts, and blocklist
    global examples, prompts, blocklist
    examples = read_examples(args.example_path)
    prompts = read_prompts(args.prompt_path)

    if args.use_blocklist:
        blocklist = read_blocklist(args.blocklist_path)
    else:
        blocklist = []
    print(f'* Using blocklist for filtering suggestions: {args.use_blocklist} ({len(blocklist)})')

    global iqa_questions, iqa_sequences
    iqa_questions = read_iqa_questions(args.iqa_question_path)
    iqa_sequences = read_iqa_sequences(
        args.iqa_sequence_path,
        args.num_questions_per_sequence
    )

    global summarization_samples, summarization_sequences
    summarization_samples = read_summarization_samples(
        args.summarization_sample_path
    )
    summarization_sequences = read_summarization_sequences(
        args.summarization_sequence_path
    )

    # Check if access codes are specified
    global allowed_access_codes
    allowed_access_codes = read_access_codes(
        args.data_dir,
        len(iqa_sequences)
    )

    global session_id_history
    session_id_history = dict()
    session_id_history = update_session_id_history(
        session_id_history,
        args.session_id_history_path
    )

    # Make log directory if not exists
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=args.debug,
    )
