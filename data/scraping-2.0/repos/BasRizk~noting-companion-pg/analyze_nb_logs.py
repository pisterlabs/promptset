
import os
from _log_parser import LogParser
from _nb_parser import NotebookParser
from _utils import (
    construct_prompt,
    prompt,
    pprint_msg,
    count_tokens_in_prompt_messages,
    get_all_file_with_extension_in_dir_recursively
)
import json
import openai
from _log_parser import LogEntry
from _nb_progress import get_notebook_progress
from _utils import Tee




import os
from _log_parser import LogParser
from _nb_parser import NotebookParser
from _utils import get_all_file_with_extension_in_dir_recursively


notebooks_dir = 'tac_notebooks'
logs_dir = 'tac_raw_logs'

all_log_filepathes = get_all_file_with_extension_in_dir_recursively(logs_dir, ".log")
all_log_filepathes.sort()
print(f'There are {len(all_log_filepathes)} log files in {logs_dir} directory')

print('\nExample selected log file:')
selected_log_filepath = all_log_filepathes[1]
print(f'Selected log file: {selected_log_filepath}')
log_parser = LogParser(selected_log_filepath).parse()
print('fetching notebooks from log file: ', notebooks_dir)
log_parser_per_notebook = log_parser.attach_notebooks(notebooks_dir, verbose=False)
print(f'\nLog parser per notebook:\n{log_parser_per_notebook}')


for nb_filepath, (nb_log_parser, nb_parser) in log_parser_per_notebook.items():
    print(f'Selected notebook: {nb_filepath} with {len(nb_parser)} cells in the selected notebook')
    print(f'Opened notebook: {nb_parser.filepath}')
    print(f'Log Parser only notebook: {nb_log_parser.filepath}')


def inject_content(assistant_msg, step_cell_id: int, change: LogEntry):
    assistant_msg = assistant_msg.copy()
    try:
        prepared_json_format = json.loads(assistant_msg['content'])
    except:
        try:
            prepared_json_format = eval(assistant_msg['content'])
        except:
            prepared_json_format = assistant_msg['content']
            
    prepared_json_format[step_cell_id]['content'] = change.get_formatted_content()
    return {'content': prepared_json_format}


for nb_filepath, (nb_log_parser, nb_parser) in log_parser_per_notebook.items():
    try:
        nb_progress = get_notebook_progress(nb_parser, nb_log_parser)
    except Exception as e:
        print('><'*50)
        print(f'Error: {e} with:')
        print(f'nb_filepath: {nb_filepath}')
        print(f'nb_log_parser: {nb_log_parser.filepath}')
        print('><'*50)
        continue
    
    nb_progress = [progress for progress in nb_progress if len(progress) > 0]
    nb_parser_filename = os.path.basename(nb_parser.filepath)
    nb_log_parser_filename = os.path.basename(nb_log_parser.filepath)
    unique_log_name = f'{nb_parser_filename}_{nb_log_parser_filename}.description_seq'
    tee = Tee(f'{unique_log_name}.txt')
    def print(*args, **kwargs):
        with tee:
            return __builtins__.print(*args, **kwargs)

    print(f'Notebook: {nb_parser.filepath}')
    print(f'Log: {nb_log_parser.filepath}')
    print(f'Number of progress steps: {len(nb_progress)}')
    print(f'Number of progress steps unrolled: {sum([len(step) for step in nb_progress])}')

    assistant_msgs = []
    prev_steps_applied = []
    for step_i, step in enumerate(nb_progress):
        step.reset()
        next_change_type = 'insert'
        # prev_msgs = [] # TODO should I reset prev_msgs upon each completed step?
        for change_i, nb_parser_with_change_applied in enumerate(step):
            # Use default prompt -- will be appended automatically by construct_prompt
            prev_msgs = []
            for prev_step, prev_response in zip(prev_steps_applied, assistant_msgs):
                prev_msgs.append({ "role": "user", "content": str(prev_step)})
                prev_msgs.append(prev_response)

            while True:
                try:
                    prompt_msgs = construct_prompt(nb_parser_with_change_applied, prev_messages=prev_msgs)
                    assistant_msgs.append(prompt(prompt_msgs))
                    break
                except openai.error.InvalidRequestError as e:
                    if len(prev_msgs) > 0:
                        prev_msgs = prev_msgs[-1:]
                    else:
                        raise e
                    
            prev_steps_applied.append(nb_parser_with_change_applied)
                    
            print('><'*50)
            # print('prompt_msgs')
            # for i, prompt_msg in enumerate(prompt_msgs):
            #     print(i)
            #     pprint_msg(prompt_msg)
                
            print(f'Step {step_i} Change @ {step.cell_id}:: {change_i}:  {next_change_type}')
            next_change_type = 'update'
            print(step.entries[change_i].print())
            print('><'*50)
            
            print(f'Step {step_i} Change {change_i} Response:')
            injected_contented_assistant_msg = inject_content(assistant_msgs[-1], step.cell_id, step.entries[change_i])
            pprint_msg(injected_contented_assistant_msg)




