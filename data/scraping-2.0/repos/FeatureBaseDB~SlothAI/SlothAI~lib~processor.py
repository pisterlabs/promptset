import ast
import re
import math
import time
import base64

from io import BytesIO

import requests
import json

import openai

from itertools import groupby

from google.cloud import vision, storage, documentai
from google.api_core.client_options import ClientOptions
from SlothAI.lib.util import random_string, random_name, get_file_extension, upload_to_storage, upload_to_storage_requests, split_image_by_height, download_as_bytes, create_audio_chunks
from SlothAI.lib.template import Template

from SlothAI.web.models import Token

from typing import Dict

import PyPDF2

from flask import current_app as app
from flask import url_for

from jinja2 import Environment

from enum import Enum

import datetime

# supress OpenAI resource warnings for unclosed sockets
import warnings
warnings.filterwarnings("ignore")

from SlothAI.web.custom_commands import random_word, random_sentence, chunk_with_page_filename, filter_shuffle
from SlothAI.web.models import User, Node, Pipeline

from SlothAI.lib.tasks import Task, process_data_dict_for_insert, transform_data, get_values_by_json_paths, box_required, validate_dict_structure, TaskState, NonRetriableError, RetriableError, MissingInputFieldError, MissingOutputFieldError, UserNotFoundError, PipelineNotFoundError, NodeNotFoundError, TemplateNotFoundError
from SlothAI.lib.database import table_exists, add_column, create_table, get_columns, featurebase_query
from SlothAI.lib.util import strip_secure_fields, filter_document, random_string

import SlothAI.lib.services as services

env = Environment()
env.globals['random_word'] = random_word
env.globals['random_sentence'] = random_sentence
env.globals['chunk_with_page_filename'] = chunk_with_page_filename
env.filters['shuffle'] = filter_shuffle

class DocumentValidator(Enum):
    INPUT_FIELDS = 'input_fields'
    OUTPUT_FIELDS = 'output_fields'

retriable_status_codes = [408, 409, 425, 429, 500, 503, 504]

processers = {}
processer = lambda f: processers.setdefault(f.__name__, f)

def process(task: Task) -> Task:
    user = User.get_by_uid(task.user_id)
    if not user:
        raise UserNotFoundError(task.user_id)
    
    pipeline = Pipeline.get(uid=task.user_id, pipe_id=task.pipe_id)
    if not pipeline:
        raise PipelineNotFoundError(pipeline_id=task.pipe_id)
    
    node_id = task.next_node()
    node = Node.get(uid=task.user_id, node_id=node_id)
    if not node:
        raise NodeNotFoundError(node_id=node_id)

    missing_field = validate_document(node, task, DocumentValidator.INPUT_FIELDS)
    if missing_field:
        raise MissingInputFieldError(missing_field, node.get('name'))

    # get tokens from service tokens
    # and process values for numbers
    _extras = {}

    for key, value in node.get('extras').items():
        # cast certain strings to other things
        if isinstance(value, str):  
            if f"[{key}]" in value:
                token = Token.get_by_uid_name(task.user_id, key)
                if not token:
                    raise NonRetriableError(f"You need a service token created for '{key}'.")
                value = token.get('value')
            # convert ints and floats from strings
            elif value.isdigit():
                # convert to int
                value = int(value)
            elif '.' in value and all(c.isdigit() or c == '.' for c in value):
                # convert it to a float
                value = float(value)

        _extras[key] = value

    # update node extras
    node['extras'] = _extras

    # template the extras off the node
    extras = evaluate_extras(node, task)

    if extras:
        task.document.update(extras)

    # get the user
    user = User.get_by_uid(uid=task.user_id)

    # if "x-api-key" in node.get('extras'):
    task.document['X-API-KEY'] = user.get('db_token')
    # if "database_id" in node.get('extras'):
    task.document['DATABASE_ID'] = user.get('dbid')

    # processer methods are responsible for adding errors to documents
    task = processers[node.get('processor')](node, task)

    # TODO, decide what to do with errors and maybe truncate pipeline
    if task.document.get('error'):
        return task

    if "X-API-KEY" in task.document.keys():
        task.document.pop('X-API-KEY', None)
    if "DATABASE_ID" in task.document.keys():
        task.document.pop('DATABASE_ID', None)

    # strip out the sensitive extras
    clean_extras(_extras, task)
    missing_field = validate_document(node, task, DocumentValidator.OUTPUT_FIELDS)
    if missing_field:
        raise MissingOutputFieldError(missing_field, node.get('name'))

    return task

@processer
def jinja2(node: Dict[str, any], task: Task) -> Task:

    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))
    if not template:
        raise TemplateNotFoundError(template_id=node.get('template_id'))

    template_text = Template.remove_fields_and_extras(template.get('text'))
    template_text = template_text.strip()

    try:
        if template_text:
            jinja_template = env.from_string(template_text)
            jinja = jinja_template.render(task.document)
    except Exception as e:
        raise NonRetriableError(f"jinja2 processor: unable to render jinja: {e}")

    try:
        jinja_json = json.loads(jinja)
        for k,v in jinja_json.items():
            task.document[k] = v
    except Exception as e:
        raise NonRetriableError(f"jinja2 processor: unable to load jinja output as JSON. Try throwing a pair of curly braces at the bottom or consider fixing this error: {e}")

    return task



@processer
def callback(node: Dict[str, any], task: Task) -> Task:
    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))
    if not template:
        raise TemplateNotFoundError(template_id=node.get('template_id'))
    
    user = User.get_by_uid(uid=task.user_id)
    if not user:
        raise UserNotFoundError(user_id=task.user_id)

    # need to rewrite to allow mapping tokens to the url template
    # alternately we could require a jinja template processor to be used in front of this to build the url
    auth_uri = task.document.get('callback_uri')

    # strip secure stuff out of the document
    document = strip_secure_fields(task.document) # returns document

    keys_to_keep = []
    if template.get('output_fields'):
        for field in template.get('output_fields'):
            for key, value in field.items():
                if key == 'name':
                    keys_to_keep.append(value)

        if len(keys_to_keep) == 0:
            data = document
        else:
            data = filter_document(document, keys_to_keep)
    else:
        data = document

    # must add node_id and pipe_id
    data['node_id'] = node.get('node_id')
    data['pipe_id'] = task.pipe_id

    try:
        headers = {'Content-Type': 'application/json'}
        resp = requests.post(auth_uri, data=json.dumps(data), headers=headers)
        if resp.status_code != 200:
            message = f'got status code {resp.status_code} from callback'
            if resp.status_code in retriable_status_codes:
                raise RetriableError(message)
            else:
                raise NonRetriableError(message)
        
    except (
        requests.ConnectionError,
        requests.HTTPError,
        requests.Timeout,
        requests.TooManyRedirects,
        requests.ConnectTimeout,
    ) as exception:
        raise RetriableError(exception)
    except Exception as exception:
        raise NonRetriableError(exception)

    return task


@processer
def info_file(node: Dict[str, any], task: Task) -> Task:
    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))

    # user
    user = User.get_by_uid(uid=task.user_id)
    uid = user.get('uid')

    # can't do anything without this input
    if 'filename' not in task.document:
        raise NonRetriableError("A 'filename' key must be present in the document. Use a string or an array of strings.")
    else:
        filename = task.document.get('filename')

    # make lists, like santa
    if isinstance(filename, str):
        # If filename is a string, convert it into a list
        filename = [filename]
        task.document['filename'] = filename

    # verify output fields steampunk style
    output_fields = template.get('output_fields')
    _break = [0,0,0,1] # size, ttl, content_type, pdf_num_pages
    for index, output_field in enumerate(output_fields):
        if "content_type" in output_field.get('name'):
            _break[2] = 1
        if "file_size_bytes" in output_field.get('name'):
            _break[0] = 1
        if "ttl" in output_field.get('name'):
            _break[1] = 1
        if 0 not in _break:
            break
    if 0 in _break:
        raise NonRetriableError("You must have 'content_type', 'file_size_bytes', and 'ttl' defined in output_fields. Optionally, you can use 'pdf_num_pages' for the number of pages in a PDFs.")
    
    # outputs
    task.document['file_size_bytes'] = []
    task.document['ttl'] = []
    task.document['pdf_num_pages'] = []
    storage_content_types = []

    # let's get to the chopper
    for index, file_name in enumerate(filename):
        # Get the file
        gcs = storage.Client()
        bucket = gcs.bucket(app.config['CLOUD_STORAGE_BUCKET'])
        blob = bucket.get_blob(f"{uid}/{file_name}")
        file_content = blob.download_as_bytes()

        # Get the content type of the file from metadata
        try:
            content_type = blob.content_type
            file_size_bytes = blob.size
        except:
            raise NonRetriableError("Can't get the size or content_type from the file in storage.")

        if content_type == "application/pdf":
            # Create a BytesIO object for the PDF content
            pdf_content_stream = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_content_stream)
            pdf_num_pages = len(pdf_reader.pages)
            task.document['pdf_num_pages'].append(pdf_num_pages)
        else:
            # we must add something even if it's not a PDF
            task.document['pdf_num_pages'].append(-1)

        storage_content_types.append(content_type)
        task.document['file_size_bytes'].append(file_size_bytes)
        task.document['ttl'].append(-1) # fix this

    # handle existing content_type in the document
    if not task.document['content_type'] or not isinstance(task.document.get('content_type'), list):
        task.document['content_type'] = storage_content_types

    return task


@processer
def split_task(node: Dict[str, any], task: Task) -> Task:
    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))
    input_fields = template.get('input_fields')
    output_fields = template.get('output_fields')
    if not input_fields:
        raise NonRetriableError("split_task processor: input fields required")
    if not output_fields:
        raise NonRetriableError("split_task processor: output fields required")

    inputs  = [n['name'] for n in input_fields]
    outputs = [n['name'] for n in output_fields] 

    batch_size = node.get('extras', {}).get('batch_size', None)

    task_service = app.config['task_service']

    # batch_size must be in extras
    if not batch_size:
        raise NonRetriableError("split_task processor: batch_size must be specified in extras!")
    
    try:
        batch_size = int(batch_size)
    except Exception as e:
        raise NonRetriableError("split_task processor: batch size must be an integer")

    # this call is currently required to update split status
    
    try:
        task_stored = task_service.fetch_tasks(task_id=task.id)
        task_stored = task_stored[0]
        task.split_status = task_stored['split_status']
    except Exception as e:
        raise NonRetriableError(f"getting task by ID but got none: {e}")
    # task.refresh_split_status()
    
    # all input / output fields should be lists of the same length to use split_task
    total_sizes = []
    for output in outputs:
        if output in inputs:
            field = task.document[output]
            if not isinstance(field, list):
                raise NonRetriableError(f"split_task processor: output fields must be list type: got {type(field)}")

            # if this task was partially process, we need to truncate the data
            # to only contain the data that hasn't been split.
            if task.split_status != -1:
                total_sizes.append(len(task.document[output][task.split_status:]))
                del task.document[output][:task.split_status]
            else:
                total_sizes.append(len(field))

        else:
            raise NonRetriableError(f"split_task processor: all output fields must be taken from input fields: output field {output} was not found in input fields.")

    if not all_equal(total_sizes):
        raise NonRetriableError("split_task processor: len of fields must be equal to re-batch a task")

    app.logger.info(f"Split Task: Task ID: {task.id}. Task Size: {total_sizes[0]}. Batch Size: {batch_size}. Number of Batches: {math.ceil(total_sizes[0] / batch_size)}. Task split status was set to {task.split_status}.")

    new_task_count = math.ceil(total_sizes[0] / batch_size)

    # split the data and re-task
    try:
        for i in range(new_task_count):

            task_stored = task_service.fetch_tasks(task_id=task.id)[0] # not safe

            if not task_service.is_valid_state_for_process(task_stored['state']):
                raise services.InvalidStateForProcess(task_stored['state'])

            batch_data = {}
            for field in outputs:
                batch_data[field] = task.document[field][:batch_size]
                del task.document[field][:batch_size]

            new_task = Task(
                id = random_string(),
                user_id=task.user_id,
                pipe_id=task.pipe_id,
                nodes=task.nodes[1:],
                document=batch_data,
                created_at=datetime.datetime.utcnow(),
                retries=0,
                error=None,
                state=TaskState.RUNNING,
                split_status=-1
            )

            # create new task and queue it      
            task_service.create_task(new_task)

            # commit status of split on original task
            task.split_status = (i + 1) * batch_size
            task_service.update_task(task_id=task.id, split_status=task.split_status)

            app.logger.info(f"Split Task: spawning task {i + 1} of projected {new_task_count}. It's ID is {new_task.id}")

    except services.InvalidStateForProcess as e:
        app.logger.warn(f"Task with ID {task.id} was being split. State was changed during that process.")
        raise e

    except Exception as e:
        app.logger.warn(f"Task with ID {task.id} was being split. An exception was raised during that process.")
        raise NonRetriableError(e)

    # the initial task doesn't make it past split_task. so remove the rest of the nodes
    task.nodes = [task.next_node()]
    return task


def sloth_embedding(input_field: str, output_field: str, model: str, task: Task) -> Task:

    data = {
        "text": task.document[input_field],
        "model": model
        # data = get_values_by_json_paths(keys, task.document)
        # TODO: could be a nested key
    }

    defer, selected_box = box_required()
    if defer:
        raise RetriableError("sloth virtual machine is being started")
    
    url = f"http://sloth:{app.config['SLOTH_TOKEN']}@{selected_box.get('ip_address')}:9898/embed"

    try:
        # Send the POST request with the JSON data
        response = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"}, timeout=60)
    except Exception as ex:
        raise NonRetriableError(f"Exception raised connecting to sloth virtual machine: {ex}")

    # Check the response status code for success
    if response.status_code == 200:
        task.document[output_field] = response.json().get("embeddings")
    else:
        raise NonRetriableError(f"Embedding server is overloaded. Error code: {response.status_code}. The likely reason is that you've asked to embed too many things at once. Try splitting your tasks.")

    return task


@processer
def embedding(node: Dict[str, any], task: Task) -> Task:
    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))
    
    extras = node.get('extras', None)
    if not extras:
        raise NonRetriableError("embedding processor: extras not found but is required")
    
    model = extras.get('model')
    if not model:
        raise NonRetriableError("embedding processor: model not found in extras but is required")

    input_fields = template.get('input_fields')
    if not input_fields:
        raise NonRetriableError("embedding processor: input_fields required.")
    
    output_fields = template.get('output_fields')
    if not output_fields:
        raise NonRetriableError("embedding processor: output_fields required.")

    # Loop through each input field and produce the proper output for each <key>_embedding output field
    for input_field in input_fields:
        input_field_name = input_field.get('name')

        # Define the output field name for embeddings
        output_field = f"{input_field_name}_embedding"

        # Check if the output field is in output_fields
        if output_field not in [field['name'] for field in output_fields]:
            raise NonRetriableError(f"'{output_field}' is not in 'output_fields'.")

        # Get the input data chunks
        input_data = task.document.get(input_field_name)

        # Initialize a list to store the embeddings
        embeddings = []

        if model == "text-embedding-ada-002":
            openai.api_key = task.document.get('openai_token')
            try:
                batch_size = 10
                for i in range(0, len(input_data), batch_size):
                    batch = input_data[i:i + batch_size]
                    embedding_results = openai.embeddings.create(input=batch, model=task.document.get('model'))
                    embeddings.extend([_object.embedding for _object in embedding_results.data])

                # Add the embeddings to the output field
                task.document[output_field] = embeddings
            except Exception as ex:
                app.logger.info(f"embedding processor: {ex}")

                raise NonRetriableError(f"Exception talking to OpenAI ada embedding: {ex}")


        elif "instructor" in model:
            task = sloth_embedding(input_field_name, output_field, model, task)

    return task


# complete strings
@processer
def aichat(node: Dict[str, any], task: Task) -> Task:
    # output and input fields
    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))
    if not template:
        raise TemplateNotFoundError(template_id=node.get('template_id'))

    # outputs
    output_fields = template.get('output_fields')
    output_field = output_fields[0].get('name') # always use the first output field

    # inputs
    input_fields = template.get('input_fields')

    # Check if each input field is present in 'task.document'
    for field in input_fields:
        field_name = field['name']
        if field_name not in task.document:
            raise NonRetriableError(f"Input field '{field_name}' is not present in the document.")

    # replace single strings with lists
    task.document = process_input_fields(task.document, input_fields)

    if "gpt" in task.document.get('model'):
        openai.api_key = task.document.get('openai_token')

        template_text = Template.remove_fields_and_extras(template.get('text'))

        if template_text:
            jinja_template = env.from_string(template_text)
            prompt = jinja_template.render(task.document)
        else:
            raise NonRetriableError("Couldn't find template text.")

        system_prompt = task.document.get('system_prompt', "You are a helpful assistant.")

        # we always reverse the lists, so it's easier to do without timestamps
        message_history = task.document.get('message_history', [])
        role_history = task.document.get('role_history', [])

        if message_history or role_history:
            if len(message_history) != len(role_history):
                raise NonRetriableError("'role_history' length must match 'message_history' length.")
            else:
                message_history.reverse()
                role_history.reverse()

        chat_messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Iterate through the user history
        for idx, message in enumerate(message_history):
            # Determine the role (user or assistant) based on the index
            role = role_history[idx]

            # Create a message object and append it to the chat_messages list
            chat_messages.append({
                "role": role,
                "content": message
            })

        # add the user input
        chat_messages.append({"role": "user", "content": prompt})

        retries = 3
        # try a few times
        for _try in range(retries):
            completion = openai.chat.completions.create(
                model = task.document.get('model'),
                messages = chat_messages
            )

            answer = completion.choices[0].message.content

            if answer:
                task.document[output_field] = answer
                return task

        else:               
            raise NonRetriableError(f"Tried {retries} times to get an answer from the AI, but failed.")    
    
    else:
        raise NonRetriableError("The aichat processor expects a supported model.")


# complete dictionaries
def ai_prompt_to_dict(model="gpt-3.5-turbo-1106", prompt="", retries=3):
    # negotiate the format
    if model == "gpt-3.5-turbo-1106" and "JSON" in prompt:
        system_content = "You write JSON for the user."
        response_format = {'type': "json_object"}
    else:
        system_content = "You write JSON dictionaries for the user, without using text markup or wrappers.\nYou output things like:\n'''ai_dict={'akey': 'avalue'}'''"
        response_format = None

    # try a few times
    for _try in range(retries):
        completion = openai.chat.completions.create(
            model = model,
            response_format = response_format,
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
        )
        ai_dict_str = completion.choices[0].message.content.replace("\n", "").replace("\t", "")
        ai_dict_str = re.sub(r'\s+', ' ', ai_dict_str).strip()
        ai_dict_str = re.sub(r'^ai_dict\s*=\s*', '', ai_dict_str)

        try:
            ai_dict = eval(ai_dict_str)
            if ai_dict.get('ai_dict'):
                ai_dict = ai_dict('ai_dict')
            err = None
            break
        except (ValueError, SyntaxError, NameError, AttributeError, TypeError) as ex:
            ai_dict = {}
            err = f"AI returned a un-evaluatable, non-dictionary object on try {_try} of {retries}: {ex}"
            time.sleep(2) # give it a few seconds

    return err, ai_dict

@processer
def aidict(node: Dict[str, any], task: Task) -> Task:
    # templates
    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))
    if not template:
        raise TemplateNotFoundError(template_id=node.get('template_id'))
    
    input_fields = template.get('input_fields')
    
    # Check if each input field is present in 'task.document'
    for field in input_fields:
        field_name = field['name']
        if field_name not in task.document:
            raise NonRetriableError(f"Input field '{field_name}' is not present in the document.")

    # replace single strings with lists
    task.document = process_input_fields(task.document, input_fields)

    # Check if there are more than one input fields and grab the iterate_field
    if len(input_fields) > 1:
        iterate_field_name = task.document.get('iterate_field')
        if not iterate_field_name:
            raise NonRetriableError("More than one input field requires an 'iterate_field' value in extras.")
        if iterate_field_name != "False" and iterate_field_name not in [field['name'] for field in input_fields]:
            raise NonRetriableError(f"'{iterate_field_name}' must be present in 'input_fields' when there are more than one input fields, or you may use a 'False' string for no iteration.")
    else:
        # use the first field
        iterate_field_name = input_fields[0]['name']

    if iterate_field_name != "False":
        iterator = task.document.get(iterate_field_name)
    else:
        iterator = ["False"]

    # Identify if the iterator is a list of lists
    is_list_of_lists = isinstance(iterator[0], list)

    # output fields
    output_fields = template.get('output_fields')

    # get the model and begin
    model = task.document.get('model')
    if "gpt" in model:
        openai.api_key = task.document.get('openai_token')

        # Loop over iterator field list, or if "False", just loop once
        for outer_index, item in enumerate(iterator):
            # if the item is not a list, we make it one
            if not isinstance(item, list):
                item = [item]

            ai_dicts = []
            
            # loop over inner list
            for inner_index in range(len(item)):
                # set the fields for the template's loop inclusions, if it has any
                task.document['outer_index'] = outer_index
                task.document['inner_index'] = inner_index

                template_text = Template.remove_fields_and_extras(template.get('text'))

                if template_text:
                    jinja_template = env.from_string(template_text)
                    prompt = jinja_template.render(task.document)
                else:
                    raise NonRetriableError("Couldn't find template text.")

                # call the ai
                err, ai_dict = ai_prompt_to_dict(
                    model=model,
                    prompt=prompt,
                    retries=3
                )
                ai_dicts.append(ai_dict)

                if err:
                    raise NonRetriableError(err)

            if is_list_of_lists:
                # Process as list of lists
                sublist_result_dict = {field['name']: [] for field in output_fields}
                for dictionary in ai_dicts:
                    for key, value in dictionary.items():
                        sublist_result_dict[key].append(value)

                for field in output_fields:
                    field_name = field['name']
                    if field_name not in task.document:
                        task.document[field_name] = []
                    task.document[field_name].append(sublist_result_dict[field_name])
            else:
                # Process as a simple list
                result_dict = {field['name']: [] for field in output_fields}
                for dictionary in ai_dicts:
                    for key, value in dictionary.items():
                        result_dict[key].append(value)

                for field in output_fields:
                    field_name = field['name']
                    if field_name not in task.document:
                        task.document[field_name] = []
                    task.document[field_name].extend(result_dict[field_name])

        task.document.pop('outer_index', None)
        task.document.pop('inner_index', None)

        return task

    else:
        raise NonRetriableError("The aidict processor expects a supported model.")


# look at a picture and get stuff
@processer
def aivision(node: Dict[str, any], task: Task) -> Task:
    # Output and input fields
    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))
    if not template:
        raise TemplateNotFoundError(template_id=node.get('template_id'))

    user = User.get_by_uid(uid=task.user_id)
    uid = user.get('uid')
    model = task.document.get('model', "gv-objects")

    # use the first output field
    try:
        output_fields = template.get('output_fields')
        output_field = output_fields[0].get('name')
    except:
        output_field = "objects"

    # Check if each input field is present in 'task.document'
    input_fields = template.get('input_fields')

    for field in input_fields:
        field_name = field['name']
        if field_name not in task.document:
            raise NonRetriableError(f"Input field '{field_name}' is not present in the document.")

    if not task.document.get('filename') or not task.document.get('content_type'):
        raise NonRetriableError("Document must contain 'filename' and 'content_type'.")

    filename = task.document.get('filename')
    content_type = task.document.get('content_type')

    # Deal with lists
    if isinstance(filename, list):
        if not isinstance(content_type, list):
            raise NonRetriableError("If filename is a list, content_type must also be a list.")
        if len(filename) != len(content_type):
            raise NonRetriableError("Document must contain equal size lists of filename and content-type.")
        filename = filename[0]
        content_type = content_type[0]
    elif isinstance(filename, str) and isinstance(content_type, str):
        # If both variables are strings, convert them into lists
        filename = [filename]
        content_type = [content_type]
    else:
        # If none of the conditions are met, raise a NonRetriableError
        raise NonRetriableError("Both filename and content_type must either be equal size lists or strings.")

    # Check if the mime type is supported for PNG, JPG, and BMP
    supported_content_types = ['image/png', 'image/jpeg', 'image/bmp', 'image/jpg']

    for index, file_name in enumerate(filename):
        content_parts = content_type[index].split(';')[0]
        if content_parts not in supported_content_types:
            raise NonRetriableError(f"Unsupported file type for {file_name}: {content_type[index]}")

    # loop through the detection filenames
    for index, file_name in enumerate(filename):
        # models are gv-objects, gpt-scene, and gv-ocr
        if model == "gv-objects":
            # Now run the code for image processing
            image_uri = f"gs://{app.config['CLOUD_STORAGE_BUCKET']}/{uid}/{file_name}"

            client = vision.ImageAnnotatorClient()
            response = client.annotate_image({
                'image': {'source': {'image_uri': image_uri}},
                'features': [{'type_': vision.Feature.Type.LABEL_DETECTION}]
            })

            # Get a list of detected labels (objects)
            labels = [label.description.lower() for label in response.label_annotations]

            # Append the labels list to task.document[output_field]
            if not task.document.get(output_field):
                task.document[output_field] = []
            task.document[output_field].append(labels)

        elif "gpt" in model:
            # Get the document
            gcs = storage.Client()
            bucket = gcs.bucket(app.config['CLOUD_STORAGE_BUCKET'])
            blob = bucket.blob(f"{uid}/{file_name}")

            # download to file
            content = blob.download_as_bytes()
            base64_data = base64.b64encode(content).decode('utf-8')

            # move this to util and clean up
            headers = {
              "Content-Type": "application/json",
              "Authorization": f"Bearer {task.document.get('openai_token')}"
            }
            if 'system_prompt' in task.document:
                system_prompt = task.document.get('system_prompt')
            else:
                system_prompt = "What is in the image?"

            payload = {
              "model": model,
              "messages": [
                {
                  "role": "user",
                  "content": [
                    {
                      "type": "text",
                      "text": system_prompt
                    },
                    {
                      "type": "image_url",
                      "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_data}"
                      }
                    }
                  ]
                }
              ],
              "max_tokens": 300
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            if 'error' in response.json():
                raise NonRetriableError(response.json().get('error').get('message'))

            # Append the labels list to task.document[output_field]
            try:
                scene = response.json().get('choices')[0].get('message').get('content')
            except:
                raise NonRetriableError("Couldn't decode a response from the AI.")

            if not task.document.get(output_field):
                task.document[output_field] = []
            task.document[output_field].append(scene)
    
        elif model == "gv-ocr":
            # Now run the code for image processing
            image_uri = f"gs://{app.config['CLOUD_STORAGE_BUCKET']}/{uid}/{file_name}"

            # Download the file contents as bytes
            content = download_as_bytes(uid,file_name)

            # Split the image into segments by height
            image_segments = split_image_by_height(BytesIO(content))

            # Initialize the Vision API client
            vision_client = vision.ImageAnnotatorClient()

            _texts = [] # array by image segment (page)

            try:
                for i, segment_bytesio in enumerate(image_segments):
                    # Create a vision.Image object for each segment
                    segment_image = vision.Image(content=segment_bytesio.read())

                    # Detect text in the segment
                    response = vision_client.document_text_detection(image=segment_image)

                    # Get text annotations for the segment
                    texts = response.text_annotations

                    # Extract only the descriptions (the actual text)
                    for text in texts:
                        # append it to our new array
                        _texts.append(text.description)
                        break

            except Exception as ex:
                raise NonRetriableError(f"Something went wrong with character detection. Contact support: {ex}")

            if not task.document.get(output_field):
                task.document[output_field] = []
            task.document[output_field].append(_texts)

        else:
            raise NonRetriableError("Vision model not found. Check extras.")

    return task


# generate images off a prompt
@processer
def aiimage(node: Dict[str, any], task: Task) -> Task:
    # Output and input fields
    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))
    if not template:
        raise TemplateNotFoundError(template_id=node.get('template_id'))

    user = User.get_by_uid(uid=task.user_id)
    uid = user.get('uid')

    output_fields = template.get('output_fields')   
    # Check that there is no more than one output field
    if len(output_fields) > 1:
        raise NonRetriableError("Only one output field is allowed in 'output_fields'.")

    # use the first output field, or set one
    try:
        output_field = output_fields[0].get('name')
    except:
        output_field = "objects"

    # Check if each input field is present in 'task.document'
    input_fields = template.get('input_fields')

    # Ensure there is only one input field
    if len(input_fields) != 1:
        raise NonRetriableError("Only one input field is allowed in 'input_fields'.")

    for field in input_fields:
        field_name = field['name']
        if field_name not in task.document:
            raise NonRetriableError(f"Input field '{field_name}' is not present in the document.")
    
    input_field = input_fields[0].get('name')

    # Get the value associated with input_field from the document
    input_value = task.document.get(input_field)

    if isinstance(input_value, str):
        # If it's a string, convert it to a list with one element
        prompts = [input_value]
    elif isinstance(input_value, list):
        # If it's a list, ensure it contains only strings
        if all(isinstance(item, str) for item in input_value):
            prompts = input_value
        else:
            raise NonRetriableError("Input field must be a string or a list of strings.")
    else:
        raise NonRetriableError("Input field must be a string or a list of strings.")

    # Apply [:1000] to the input field as the prompt
    if not task.document.get(output_field):
        task.document[output_field] = []

    for prompt in prompts:
        prompt = prompt[:1000]

        if not prompt:
            raise NonRetriableError("Input field is required and should contain the prompt.")

        num_images = task.document.get('num_images', 0)
        if not num_images:
            num_images = 1

        if "dall-e" in task.document.get('model'):
            openai.api_key = task.document.get('openai_token')

            try:
                response = openai.images.generate(
                    prompt=prompt,
                    model=task.document.get('model'),
                    n=int(num_images),
                    size="1024x1024"
                )

                urls = []
                revised_prompts = []

                # Loop over the 'data' list and extract the 'url' from each item
                for item in response.data:
                    if item.url:
                        urls.append(item.url)

                task.document[output_field].append(urls)

            except Exception as ex:
                # non-retriable error for now but add retriable as needed
                raise NonRetriableError(f"aiimage processor: exception talking to OpenAI image create: {ex}")
        else:
            raise NonRetriableError(f"Need a valid model. Try 'dall-e-2' or 'dall-e-3'.")

    return task


@processer
def read_file(node: Dict[str, any], task: Task) -> Task:
    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))
    if not template:
        raise TemplateNotFoundError(template_id=node.get('template_id'))

    user = User.get_by_uid(uid=task.user_id)
    uid = user.get('uid')

    output_fields = template.get('output_fields')   
    # Check that there no more than one output field
    if len(output_fields) > 1:
        raise NonRetriableError("Only one output field is allowed in 'output_fields'.")

    # use the first output field, or set one
    try:
        output_field = output_fields[0].get('name')
    except:
        output_field = "texts"

    filename = task.document.get('filename')
    content_type = task.document.get('content_type')

    # Deal with lists
    if isinstance(filename, list):
        if not isinstance(content_type, list):
            raise NonRetriableError("If filename is a list, content_type must also be a list.")
        if len(filename) != len(content_type):
            raise NonRetriableError("Document must contain equal size lists of filename and content-type.")
    elif isinstance(filename, str) and isinstance(content_type, str):
        # If both variables are strings, convert them into lists
        filename = [filename]
        content_type = [content_type]
    else:
        # If none of the conditions are met, raise a NonRetriableError
        raise NonRetriableError("Both filename and content_type must either be equal size lists or strings.")

    # Check if the mime type is supported
    supported_content_types = ['application/pdf', 'text/plain']

    for index, file_name in enumerate(filename):
        content_parts = content_type[index].split(';')[0]
        if content_parts not in supported_content_types:
            raise NonRetriableError(f"Unsupported file type for {file_name}: {content_type[index]}")
    
    # set page_numbers from page_number array
    page_numbers = task.document.get('page_number')

    # Process page numbers
    if page_numbers is not None:
        if not isinstance(page_numbers, list):
            raise NonRetriableError("Page numbers must be a list.")
        
        # Check if the number of page number lists is the same as the number of filenames
        if len(page_numbers) != len(filename):
            raise NonRetriableError("Filename and page numbers lists must be of the same size.")

    # set the output field
    if not task.document.get(output_field):
        task.document[output_field] = []

    # loop over the filenames
    for index, file_name in enumerate(filename):
        if "application/pdf" in content_type[index]:
            # Get the document
            gcs = storage.Client()
            bucket = gcs.bucket(app.config['CLOUD_STORAGE_BUCKET'])
            blob = bucket.blob(f"{uid}/{file_name}")
            image_content = blob.download_as_bytes()

            # Create a BytesIO object for the PDF content
            pdf_content_stream = BytesIO(image_content)
            pdf_reader = PyPDF2.PdfReader(pdf_content_stream)

            index_pages = []

            num_pdf_pages = len(pdf_reader.pages)

            # build an alernate list of page numbers from the pipeline
            if page_numbers:
                    if page_numbers[index] < 1:
                        raise NonRetriableError("Page numbers must be whole numbers > 0.")
                    if page_numbers[index] > num_pdf_pages:
                        raise NonRetriableError(f"Page number ({page_numbers[index]}) is larger than the number of pages ({num_pdf_pages}).")   
                    index_pages.append(page_numbers[index]-1)
            else:
                for page_number in range(num_pdf_pages):
                    index_pages.append(page_number)

            # processor for document ai
            opts = ClientOptions(api_endpoint=f"us-documentai.googleapis.com")
            client = documentai.DocumentProcessorServiceClient(client_options=opts)
            parent = client.common_location_path(app.config['PROJECT_ID'], "us")
            processor_list = client.list_processors(parent=parent)
            for processor in processor_list:
                name = processor.name
                break # stupid google objects

            pdf_processor = client.get_processor(name=name)

            texts = []

            # build seperate pages and process each, adding text to texts
            for page_num in index_pages:
                pdf_writer = PyPDF2.PdfWriter()
                pdf_writer.add_page(pdf_reader.pages[page_num])
                page_stream = BytesIO()
                pdf_writer.write(page_stream)

                # Get the content of the current page as bytes
                page_content = page_stream.getvalue()

                # load data
                raw_document = documentai.RawDocument(content=page_content, mime_type="application/pdf")

                # make request
                request = documentai.ProcessRequest(name=pdf_processor.name, raw_document=raw_document)
                result = client.process_document(request=request)
                document = result.document

                # move to texts
                texts.append(document.text.replace("'","`").replace('"', '``').replace("\n"," ").replace("\r"," ").replace("\t"," "))

                # Close the page stream
                page_stream.close()

        elif "text/plain" in content_type[index]:
            # grab document
            gcs = storage.Client()
            bucket = gcs.bucket(app.config['CLOUD_STORAGE_BUCKET'])
            blob = bucket.blob(f"{uid}/{file_name}")
            text = blob.download_as_text()

            # split on words
            words = text.split()
            chunks = []
            current_chunk = []

            # set the page chunk size (number of characters per page)
            page_chunk_size = task.document.get('page_chunk_size', 1536)

            # build the chunks
            for word in words:
                current_chunk.append(word)
                if len(current_chunk) >= page_chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []

            # append any leftovers
            if current_chunk:
                chunks.append(' '.join(current_chunk))

            texts = chunks

        else:
            raise NonRetriableError("Processor read_file supports text/plain and application/pdf content types only.")

        # update the document
        task.document[output_field].extend(texts)
    
    return task


@processer
def read_uri(node: Dict[str, any], task: Task) -> Task:
    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))
    
    # OpenAI only for now
    openai.api_key = task.document.get('openai_token')

    if not template:
        raise TemplateNotFoundError(template_id=node.get('template_id'))

    user = User.get_by_uid(uid=task.user_id)
    uid = user.get('uid')
    
    # use the first output field TODO FIX THIS
    try:
        output_fields = template.get('output_fields')
        output_field = output_fields[0].get('name')
    except:
        output_field = "texts"

    input_fields = template.get('input_fields')
    if not input_fields:
        raise NonRetriableError("Input fields required for the read_uri processor.")

    uri = task.document.get('uri')[0] # get the first uri
    method = task.document.get('method')

    if not uri or not method:
        raise NonRetriableError("URI and method are required in the input fields.")

    # scan for required fields in the input
    for field in input_fields:
        field_name = field['name']
        if field_name not in task.document:
            raise NonRetriableError(f"Input field '{field_name}' is not present in the document.")

    # Assuming 'bearer_token' is also a part of input_fields when needed
    bearer_token = task.document.get('bearer_token')

    # Now, you can proceed to build the request based on the method and URI
    if method == 'GET':
        # Perform a GET request
        if bearer_token:
            headers = {'Authorization': f'Bearer {bearer_token}'}
            response = requests.get(uri, headers=headers)
        else:
            response = requests.get(uri)

    elif method == 'POST':
        data = {}
        for field in task.document.get('data_fields'):
            data[field] = task.document[field]
        
        # Perform a POST request
        if bearer_token:
            headers = {'Authorization': f'Bearer {bearer_token}'}
            response = requests.post(uri, headers=headers, json=data, stream=True, allow_redirects=True)

        else:
            response = requests.post(uri, json=data, stream=True, allow_redirects=True)

    else:
        raise NonRetriableError("Request must contain a 'method' key that is one of: ['GET','POST'].")

    if response.status_code != 200:
        raise NonRetriableError(f"Request failed with status code: {response.status_code}")

    # Check if the Content-Disposition header is present
    if 'Content-Type' in response.headers:
        content_type = response.headers['Content-Type']
    else:
        content_type = tasks.document.get('content_type')
        if not content_type:
            raise NonRetriableError("This URL will require using the filename and content_type fields.")

    if 'Content-Type' in response.headers:
        content_type = response.headers['Content-Type']
    else:
        content_type = task.document.get('content_type')

    filename = task.document.get('filename')

    if filename is None and content_type is None:
        raise NonRetriableError("This URL will require using the filename and content_type fields.")

    if filename is None:
        file_extension = get_file_extension(content_type)

        if not file_extension:
            raise NonRetriableError("This URL will require using the filename and content_type fields.")
        filename = f"{random_string(16)}.{file_extension}"

    bucket_uri = upload_to_storage_requests(uid, filename, response.content, content_type)

    task.document['filename'] = filename
    task.document['content_type'] = content_type

    return task


@processer
def aispeech(node: Dict[str, any], task: Task) -> Task:
    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))

    # OpenAI only for now
    openai.api_key = task.document.get('openai_token')

    if not template:
        raise TemplateNotFoundError(template_id=node.get('template_id'))

    output_fields = template.get('output_fields')
    input_fields = template.get('input_fields')

    # needs to be moved
    # scan for required fields in the input
    for field in input_fields:
        field_name = field['name']
        if field_name not in task.document:
            raise NonRetriableError(f"Input field '{field_name}' is not present in the document.")

    # stuff this into util.py TODO
    f = False
    for index, output_field in enumerate(output_fields):
        if "uri" in output_field.get('name'):
            f = True
            break
    else:
        raise NonRetriableError("You must have 'uri' defined in output_fields. This will return URL(s) for retrieving the audio file(s).")

    # voice and model
    voice = task.document.get('voice')
    model = task.document.get('model')
    if not voice:
        voice = "alloy"
    if not model:
        model = "tts-1"
    
    # user stuff, arguable we need it
    user = User.get_by_uid(uid=task.user_id)
    uid = user.get('uid')

    # grab the first input field name that isn't the filename
    uses_filename = False
    for _input_field in input_fields:
        if 'filename' in _input_field.get('name'):
            uses_filename = True
            continue
        else:
            input_field = _input_field.get('name')

    # set the filename from the document
    if uses_filename:
        filename = task.document.get('filename')
        if not filename:
            raise NonRetriableError("If you specify an input field for filename, you must provide one for use in this node.")
        if isinstance(filename, list):
            filename = filename[0]
    else:
        rand_name = random_name(2)
        filename = f"{rand_name}-mitta-aispeech.mp3"

    # lets support only a list of strings
    if isinstance(task.document.get(input_field), list):
        items = task.document.get(input_field)
        if isinstance(items[0], list):
            raise NonRetriableError(f"Input field {input_field} needs to be a list of strings. Found a list of lists instead.")
    else:
        items = [task.document.get(input_field)]

    task.document['uri'] = []
    
    # loop through items to process
    for index, item in enumerate(items):
        response = openai.audio.speech.create(
            model = model,
            voice = voice,
            input = item
        )

        # Example usage
        new_filename = add_index_to_filename(filename, index)
        
        # Assuming 'response.content' contains the binary content of the audio file
        audio_data = response.content

        content_type = 'audio/mpeg'  # Set the appropriate content type for the audio file
        bucket_uri = upload_to_storage_requests(uid, new_filename, audio_data, content_type)

        access_uri = f"https://{app.config.get('APP_DOMAIN')}/d/{user.get('name')}/{new_filename}?token="
        task.document['uri'].append(access_uri)

    return task


@processer
def aiaudio(node: Dict[str, any], task: Task) -> Task:
    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))

    # OpenAI only for now
    openai.api_key = task.document.get('openai_token')

    if not template:
        raise TemplateNotFoundError(template_id=node.get('template_id'))
    
    # use the first output field
    try:
        output_fields = template.get('output_fields')
        output_field = output_fields[0].get('name')
    except:
        output_field = "texts"

    user = User.get_by_uid(uid=task.user_id)
    uid = user.get('uid')
    filename = task.document.get('filename')
    content_type = task.document.get('content_type')

    # Check if the mime type is supported
    supported_content_types = ['audio/mpeg', 'audio/mpeg3', 'audio/x-mpeg-3', 'audio/mp3', 'audio/mpeg-3', 'audio/wav', 'audio/webm', 'audio/mp4', 'audio/x-m4a', 'audio/m4a', 'audio/x-wav']
    
    for supported_type in supported_content_types:
        if supported_type in content_type:
            break
    else:
        raise NonRetriableError(f"Unsupported file type: {content_type}")

    # Get the document
    gcs = storage.Client()
    bucket = gcs.bucket(app.config['CLOUD_STORAGE_BUCKET'])
    blob = bucket.blob(f"{uid}/{filename}")
    audio_file = BytesIO()

    audio_file.name = f"{filename}"  # You can choose a unique name
    
    # download to file
    blob.download_to_file(audio_file)
    audio_file.content_type = content_type
    
    # must seek to the start
    audio_file.seek(0)

    # Get the file size using the BytesIO object's getbuffer method
    file_size = len(audio_file.getbuffer())

    # Check file size limit (25 MB)
    max_original_file_size = 25 * 1024 * 1024  # 25 MB in bytes
    if file_size > max_original_file_size:
        raise NonRetriableError("Original file size exceeds the 25 MB limit.")

    # removed this for now due to 32MB file size limit on appengine
    # Process the audio: split the audio into chunks
    # audio_chunks = create_audio_chunks(audio_file) # this might work if you use it

    # for now, just do one up to 25MB
    audio_chunks = [audio_file]

    # Iterate through each chunk and send it to Whisper
    for chunk_stream in audio_chunks:
        # process the audio
        model = task.document.get('model', "whisper-1")
        transcript = openai.audio.transcriptions.create(model=model, file=chunk_stream)

        # split on words
        words = transcript.text.split(" ")
        chunks = []
        current_chunk = []

        # set the page chunk size (number of characters per page)
        page_chunk_size = int(task.document.get('page_chunk_size', 1536))

        # build the chunks
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= page_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []

        # append any leftovers
        if current_chunk:
            chunks.append(' '.join(current_chunk))

    # list of strings
    task.document[output_field] = chunks
        
    # Return the modified task
    return task


@processer
def read_fb(node: Dict[str, any], task: Task) -> Task:
    user = User.get_by_uid(task.user_id)

    # if the user has a dbid, then use their database
    if user.get('dbid'):    
        doc = {
            "dbid": user.get('dbid'),
            "db_token": user.get('db_token'),
            "sql": task.document['sql']
        }
    else:
        # use a shared database
        if task.document.get('table'):
            try:
                # swap the table name on the fly
                table = task.document.get('table')
                sql = task.document.get('sql')
                username = user.get('name')
                pattern = f'\\b{table}\\b'
                sql = re.sub(pattern, f"{username}_{table}", sql)

                doc = {
                    "dbid": app.config['SHARED_FEATUREBASE_ID'],
                    "db_token": app.config['SHARED_FEATUREBASE_TOKEN'],
                    "sql": sql
                }
            except:
                raise NonRetriableError("Unable to template your SQL to the shared database. Check syntax.")
        else:
            raise NonRetriableError("Specify a 'table' key and value and template the table in your SQL.")

    resp, err = featurebase_query(document=doc)
    if err:
        if "exception" in err:
            raise RetriableError(err)
        else:
            # good response from the server but query error
            raise NonRetriableError(err)

    # response data
    fields = []
    data = {}

    for field in resp.schema['fields']:
        fields.append(field['name'])
        data[field['name']] = []

    for tuple in resp.data:
        for i, value in enumerate(tuple):
            data[fields[i]].append(value)

    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))
    if not template:
        raise TemplateNotFoundError(template_id=node.get('template_id'))
    
    _keys = template.get('output_fields')
    if _keys:
        keys = [n['name'] for n in _keys]
        task.document.update(transform_data(keys, data))
    else:
        task.document.update(data)

    return task


from SlothAI.lib.schemar import Schemar
@processer
def write_fb(node: Dict[str, any], task: Task) -> Task:
    user = User.get_by_uid(task.user_id)

    # if the user has a dbid, then use their database
    if user.get('dbid'):    
        # auth = {"dbid": task.document['DATABASE_ID'], "db_token": task.document['X-API-KEY']}
        auth = {
            "dbid": user.get('dbid'),
            "db_token": user.get('db_token')
        }
        table = task.document['table']
    else:
        # use a shared database
        if task.document.get('table'):
            auth = {
                "dbid": app.config['SHARED_FEATUREBASE_ID'],
                "db_token": app.config['SHARED_FEATUREBASE_TOKEN']
            }
        else:
            raise NonRetriableError("Specify a 'table' key and value and template the table in your SQL.")
        table = f"{user.get('name')}_{task.document.get('table')}"

    # create template service
    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))
    _keys = template.get('input_fields') # must be input fields but not enforced
    keys = [n['name'] for n in _keys]
    data = get_values_by_json_paths(keys, task.document)

    # check table
    tbl_exists, err = table_exists(table, auth)
    if err:
        raise NonRetriableError("Can't connect to database. Check your FeatureBase connection.")
    
    # if it doesn't exists, create it
    if not tbl_exists:
        create_schema = Schemar(data=data).infer_create_table_schema() # check data.. must be lists
        err = create_table(table, create_schema, auth)
        if err:
            if "already exists" in err:
                # between checking if the table existed and trying to create the
                # table, the table was created.
                pass
            elif "exception" in err:
                # issue connecting to FeatureBase cloud
                raise RetriableError(err)
            else:
                # good response from the server but there was a query error.
                raise NonRetriableError(f"FeatureBase returned: {err}. Check your fields are valid with a callback.")
            
    # get columns from the table
    column_type_map, task.document['error'] = get_columns(table, auth)
    if task.document.get("error", None):
        raise Exception("unable to get columns from table in FeatureBase cloud")

    columns = [k for k in column_type_map.keys()]

    # add columns if data key cannot be found as an existing column
    for key in data.keys():
        if key not in columns:
            if not task.document.get("schema", None):
                task.document['schema'] = Schemar(data=data).infer_schema()

            err = add_column(table, {'name': key, 'type': task.document["schema"][key]}, auth)
            if err:
                if "exception" in err:
                    raise RetriableError(err)
                else:
                    # good response from the server but query error
                    raise NonRetriableError(err)

            column_type_map[key] = task.document["schema"][key]

    columns, records = process_data_dict_for_insert(data, column_type_map, table)

    sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES {','.join(records)};"

    _, err = featurebase_query({"sql": sql, "dbid": auth.get('dbid'), "db_token": auth.get('db_token')})
    if err:
        if "exception" in err:
            raise RetriableError(err)
        else:
            # good response from the server but query error
            raise NonRetriableError(err)
    
    return task


# helper functions
# ================
def process_input_fields(task_document, input_fields):
    updated_document = task_document
    
    for field in input_fields:
        field_name = field['name']
        if field_name in updated_document:
            # Check if the field is present in the document
            value = updated_document[field_name]
            if not isinstance(value, list):
                # If it's not already a list, replace it with a list containing the value
                updated_document[field_name] = [value]
    
    return updated_document


# Function to encode the image
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')


def validate_document(node, task: Task, validate: DocumentValidator):
    template_service = app.config['template_service']
    template = template_service.get_template(template_id=node.get('template_id'))
    fields = template.get(validate)
    if fields:
        missing_key = validate_dict_structure(template.get('input_fields'), task.document)
        if missing_key:
            return missing_key
    
    return None


def evaluate_extras(node, task) -> Dict[str, any]:
    # get the node's current extras, which may be templated
    extras = node.get('extras', {})

    # combine with inputs
    combined_dict = extras.copy()
    combined_dict.update(task.document)

    user = User.get_by_uid(task.user_id)
    combined_dict['username'] = user.get('name')

    # eval the extras from inputs_fields first
    extras_template = env.from_string(str(combined_dict))
    extras_from_template = extras_template.render(combined_dict)
    extras_eval = ast.literal_eval(extras_from_template)

    # remove the keys that were in the document
    extras_eval = {key: value for key, value in extras_eval.items() if key not in task.document}

    return extras_eval


def add_index_to_filename(filename, index):
    name, ext = filename.rsplit('.', 1)
    return f"{name}_{index}.{ext}"


def clean_extras(extras: Dict[str, any], task: Task):
    if extras:
        for k in extras.keys():
            if k in task.document.keys():
                del task.document[k]
    return task


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)