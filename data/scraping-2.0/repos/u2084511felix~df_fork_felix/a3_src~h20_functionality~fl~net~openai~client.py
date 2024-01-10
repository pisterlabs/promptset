# -*- coding: utf-8 -*-
"""
---

title:
    "OpenAI HTTP API integration support module."

description:
    "This Python module is designed to interact
    with the OpenAI API using a separate process
    for handling requests and responses. It uses
    the multiprocessing and queue libraries to
    manage inter-process communication, and the
    openai library to interact with the OpenAI
    API. The module defines three main functions:
    coro, _daemon, and _process_request.

    coro(cfg_client) is a coroutine function that
    acts as a bridge between the main process and
    the separate process running the OpenAI API
    client. It takes a configuration dictionary
    as input, initializes multiprocessing queues
    for requests, results, and errors, and starts
    the daemon process. It then enters a loop,
    yielding results and errors to the calling
    context while accepting new requests and
    sending them to the daemon process.

    _daemon(cfg) is the main function running in
    the separate process. It takes a configuration
    dictionary as input and services the request
    queue by forwarding requests to the OpenAI
    API. It processes each request, enqueues
    results and errors onto their respective
    queues, and sleeps for a specified interval
    if no request is available.

    _process_request(request, default) is a
    helper function that processes a single
    request, returning a dictionary with the
    original request, response, and error
    information. It first builds a dictionary
    of keyword arguments for the OpenAI API call,
    using default values for any unspecified
    parameters. It then calls the OpenAI API
    with the constructed keyword arguments,
    catches any errors, and returns the result
    dictionary containing the original request,
    the API response, and any error encountered.

    Here is an example of a result data structure:

    {'type':        'openai_result'
     'error':       None,
     'state':       {},
     'request':     {'engine':      '<OPENAI_MODEL_ID>',
                     'max_tokens':  100,
                     'n':           1,
                     'prompt':      '<PROMPT_TEXT>.',
                     'stop':        None,
                     'temperature': 0.5},
     'response':    {'choices':     [{'finish_reason':  'stop',
                                      'index':          0,
                                      'logprobs':       None,
                                      'text':           '<RESPONSE_TEXT>'}],
                     'created':    1682122855,
                     'id':         'cmpl-77vHj30k0S1k8pqz2lHUueiQDMSZj',
                     'model':      '<OPENAI_MODEL_ID>',
                     'object':     'text_completion',
                     'usage':      {'completion_tokens':   57,
                                    'prompt_tokens':       9,
                                    'total_tokens':        66}}}"


id:
    "555345dc-41c5-4810-a4fc-7ccbcde432af"

type:
    dt003_python_module

validation_level:
    v00_minimum

protection:
    k00_general

copyright:
    "Copyright 2023 William Payne"

license:
    "Licensed under the Apache License, Version
    2.0 (the License); you may not use this file
    except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed
    to in writing, software distributed under
    the License is distributed on an AS IS BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
    either express or implied. See the License
    for the specific language governing
    permissions and limitations under the
    License."

...
"""


import importlib
import logging
import multiprocessing
import queue
import re
import string
import time

import openai

import fl.util


# -----------------------------------------------------------------------------
@fl.util.coroutine
def coro_workflow_handler(cfg, request_handler, template_handler):
    """
    Yield results for workflow coroutines sent to the OpenAI web API.

    """

    # Configure logging for the workflow handling coroutine.
    #
    id_system       = cfg['id_system']
    id_node         = cfg['id_node']
    level_log_event = cfg['level_log']
    if id_system and id_node:
        id_log_event = '{sys}.{node}.workflow'.format(sys  = id_system,
                                                      node = id_node)
    else:
        id_log_event = 'openai.workflow'
    (log_event, handler_log_event) = fl.log.event.logger(
                                                    str_id = id_log_event,
                                                    level  = level_log_event)

    map_workflow  = dict()  # id_workflow -> workflow data structure
    list_to_api   = list()
    list_from_api = list()

    while True:

        list_to_api.clear()
        (list_to_api, unix_time) = yield list_from_api

        # Update the workflow table.
        #
        for item in list_to_api:

            try:
                id_type = item['type']['id']
            except KeyError:
                msg = 'Badly formed prompt or parameter message. ' \
                      'Could not find type id information.'
                log_event.error(msg)
                raise RuntimeError(msg)

            if id_type == 'prompt_workflow':
                uid_workflow               = item['uid_workflow']
                map_workflow[uid_workflow] = _ensure_init_workflow(cfg, item)

        # Run each workflow in turn and
        # build up the result and error
        # lists.
        #
        list_from_api.clear()
        for (uid_workflow, workflow) in map_workflow.items():
            list_filt = list(item for item in list_to_api
                                    if item['uid_workflow'] == uid_workflow)
            list_from_api.extend(template_handler.send(
                                        (workflow['coroutine'].send(list_filt),
                                         unix_time)))

        # Attempt to send any available event log
        # line items to the rest of the system.
        #
        if handler_log_event.list_event:
            list_from_api.extend(handler_log_event.list_event)
            handler_log_event.list_event.clear()


# -----------------------------------------------------------------------------
def _ensure_init_workflow(cfg, workflow):
    """
    Ensure that the specified workflow is imported and primed.

    """
    if workflow['coroutine'] is not None:
        return workflow

    str_spec = workflow['spec']

    # If the spec is an importable string
    has_whitespace = re.search(r'\s', str_spec)
    is_import_spec = not has_whitespace
    if is_import_spec:
        (name_module, name_fcn) = str_spec.rsplit(".", 1)
        module                  = importlib.import_module(name = name_module)
        coroutine_function      = getattr(object = module, name = name_fcn)
    else:
        namespace = dict()
        exec(str_spec, namespace)
        coroutine_function = namespace['coro']

    workflow['coroutine'] = coroutine_function(cfg)
    workflow['coroutine'].send(None)  # Prime the coroutine
    return workflow


# -----------------------------------------------------------------------------
@fl.util.coroutine
def coro_template_handler(cfg, request_handler):
    """
    Yield results for templates/parameters sent to the OpenAI web API.

    Initialize 'database' of prompt templates.

    Prompt templates are persisted
    in memory so they don't need to
    be provided anew at each time step.

    It is envisaged that the templates
    will be updated from time to time
    as part of a continuous improvement
    process.

    """

    # Configure logging for the template handling coroutine.
    #
    id_system       = cfg['id_system']
    id_node         = cfg['id_node']
    level_log_event = cfg['level_log']
    if id_system and id_node:
        id_log_event = '{sys}.{node}.template'.format(sys  = id_system,
                                                      node = id_node)
    else:
        id_log_event = 'openai.template'
    (log_event, handler_log_event) = fl.log.event.logger(
                                                    str_id = id_log_event,
                                                    level  = level_log_event)

    map_template  = dict()  # id_prompt -> dict with template.
    list_request  = list()
    list_to_api   = list()
    list_from_api = list()

    while True:

        list_to_api.clear()
        (list_to_api, unix_time) = yield list_from_api
        list_from_api.clear()

        list_request.clear()
        for item in list_to_api:

            # Determine the type of the item.
            #
            try:
                id_type = item['type']['id']
            except KeyError:
                msg = 'Badly formed prompt or parameter message. ' \
                      'Could not find type id information.'
                log_event.error(msg)
                raise RuntimeError(msg)

            # Build a request from prompt params.
            #
            if id_type == 'prompt_params':

                try:
                    uid_template = item['uid_template']
                except KeyError:
                    msg = 'Badly formed parameter message. ' \
                          'Could not find template id information.'
                    log_event.error(msg)
                    raise RuntimeError(msg)

                try:
                    template = map_template[uid_template]
                except KeyError:
                    msg = 'Could not find template id: {id}.'.format(
                                                        id = uid_template)
                    log_event.error(msg)
                    raise RuntimeError(msg)

                list_request.append(_build_request(template = template,
                                                   param    = item))

            # Update map_map_template with new
            # and updated templates. Any old
            # templates with the same id_prompt
            # get overwritten.
            #
            elif id_type == 'prompt_template':

                try:
                    uid_template = item['uid_template']
                except KeyError:
                    msg = 'Badly formed template message. ' \
                          'Could not find template id information.'
                    log_event.error(msg)
                    raise RuntimeError(msg)
                else:
                    map_template[uid_template] = item

            # id_type not in (prompt_params,
            # prompt_template), logic error.
            #
            else:

                msg = 'Did not recognize id_type: {id}. '.format(id = id_type)
                log_event.error(msg)
                raise RuntimeError(msg)

        list_from_api = request_handler.send((list_request, unix_time))

        # Attempt to send any available event log
        # line items to the rest of the system.
        #
        if handler_log_event.list_event:
            list_from_api.extend(handler_log_event.list_event)
            handler_log_event.list_event.clear()



# -----------------------------------------------------------------------------
def _build_request(template, param):
    """
    Return a request dict.

    Firstly, we create a skeleton request
    using kwargs taken from the template
    data structure, and overriding them
    with kwargs taken from the param data
    structure.

    param takes priority, followed by
    template, followed by the defaults
    specified in the config.

    TODO: Defaults specified in the config
          could be applied here as well
          instead of in the daemon.

    Then fill in the template as needed
    for each OpenAI request type.

    """
    id_endpoint = template['id_endpoint']
    request = dict(**template['kwargs_req'])
    request.update(**param['kwargs_req'])
    request['id_endpoint'] = id_endpoint
    request['state']       = param['state']

    # completions --------------------------------
    if id_endpoint == 'completions':
        request['prompt'] = template['prompt'].format(
                                                **param['kwargs_tmpl'])

    # chat_completions ---------------------------
    if id_endpoint == 'chat_completions':
        request['messages'] = list()
        for tmpl_msg in template['messages']:
            req_msg = {
                'role':    tmpl_msg['role'],
                'content': tmpl_msg['content'].format(
                                                **param['kwargs_tmpl'])}
            if 'name' in tmpl_msg:
                req_msg['name'] = tmpl_msg['name']
            request['messages'].append(req_msg)

    # edits --------------------------------------
    if id_endpoint == 'edits':
        request['input']       = template['input'].format(
                                                **param['kwargs_tmpl'])
        request['instruction'] = template['instruction'].format(
                                                **param['kwargs_tmpl'])

    # images_generations -------------------------
    if id_endpoint == 'images_generations':
        request['prompt'] = template['prompt'].format(
                                                **param['kwargs_tmpl'])

    # images_edits -------------------------------
    if id_endpoint == 'images_edits':
        request['prompt'] = template['prompt'].format(
                                                **param['kwargs_tmpl'])

    # images_variations --------------------------
    if id_endpoint == 'images_variations':
        pass # No text prompt

    # embeddings ---------------------------------
    if id_endpoint == 'embeddings':
        request['input'] = template['input'].format(
                                                **param['kwargs_tmpl'])

    # audio_transcriptions -----------------------
    if id_endpoint == 'audio_transcriptions':
        request['prompt'] = template['prompt'].format(
                                                **param['kwargs_tmpl'])

    # audio_translations -------------------------
    if id_endpoint == 'audio_translations':
        request['prompt'] = template['prompt'].format(
                                                **param['kwargs_tmpl'])

    return request


# -----------------------------------------------------------------------------
@fl.util.coroutine
def coro_request_handler(cfg):
    """
    Yield results for requests sent to the OpenAI web API.

    For each list of request items that are
    sent to this coroutine, it will yield a
    tuple containing a list of result items
    followed by a list of errors encountered.

    This coroutine is responsible
    for inter process communication
    with the OpenAI client daemon
    process.

    """

    # Validate configuration data.
    #
    set_key_expected = set(('api_key',
                            'secs_interval',
                            'is_bit',
                            'is_async',
                            'default',
                            'id_system',
                            'id_node',
                            'level_log'))
    set_key_actual  = set(cfg.keys())
    set_key_missing = set_key_expected - set_key_actual
    if set_key_missing:
        raise RuntimeError('Missing key(s): {missing}'.format(
                                        missing = ', '.join(set_key_missing)))

    # Configure logging for the request handling coroutine.
    #
    id_system       = cfg['id_system']
    id_node         = cfg['id_node']
    level_log_event = cfg['level_log']
    if id_system and id_node:
        id_log_event = '{sys}.{node}.request'.format(sys  = id_system,
                                                     node = id_node)
    else:
        id_log_event = 'openai.request'
    (log_event, handler_log_event) = fl.log.event.logger(
                                                    str_id = id_log_event,
                                                    level  = level_log_event)

    # Configure queues and start the client in a
    # separate process.
    #
    if cfg['is_async']:
        queue_to_api          = multiprocessing.Queue()  # coro --> API daemon
        queue_from_api        = multiprocessing.Queue()  # API daemon --> coro
        cfg['queue_to_api']   = queue_to_api
        cfg['queue_from_api'] = queue_from_api
        daemon = multiprocessing.Process(target = _daemon_main,
                                         args   = (cfg,),
                                         name   = 'openai-client',
                                         daemon = True)
        daemon.start()

    # Loop forever, sending data to and from the
    # openai client daemon via the two
    # multiprocessing queues.
    #
    # At each step, yield results and event log
    # items back to the controlling context, and
    # get any new inbound requests to send to the
    # openai client daemon.
    #
    list_to_api   = list()
    list_from_api = list()

    while True:

        list_to_api.clear()
        (list_to_api, unix_time) = yield (list_from_api)
        list_from_api.clear()

        # Enqueue any newly submitted requests to
        # send to the daemon process, then dequeue
        # any newly returned results and event log
        # line items.
        #
        if cfg['is_async']:
            for item in list_to_api:
                try:
                    queue_to_api.put(item, block = False)
                except queue.Full as err:
                    log_event.error('Item dropped: queue_to_api is full.')
                    break

            while True:
                try:
                    item_to_api = queue_from_api.get(block = False)
                except queue.Empty:
                    break
                else:
                    item_to_api['unix_time'] = unix_time
                    list_from_api.append(item_to_api)

        # Synchronously process one request at a
        # time, filling list_from_api with results
        # and event log line items as we go.
        #
        else:
            for item_to_api in list_to_api:
                item_to_api['unix_time'] = unix_time
                list_from_api.extend(
                            _process_one_request(request_raw = item_to_api,
                                                 default     = cfg['default'],
                                                 is_bit      = cfg['is_bit'],
                                                 log_event   = log_event))

        # Attempt to send any available event log
        # line items to the rest of the system.
        #
        if handler_log_event.list_event:
            list_from_api.extend(handler_log_event.list_event)
            handler_log_event.list_event.clear()


# -----------------------------------------------------------------------------
def _daemon_main(cfg):
    """
    Service the request queue, forwarding requests to the OpenAI API.

    Each request in the queue is processed,
    pausing for secs_interval if no request
    is available at that time.

    Results and errors are enqueued onto
    their respective queues for further
    processing by the controlling process.

    This is intended to be a minimal daemon
    that sits in a separate process and
    handles communication with the OpenAI
    API.

    """

    # Configure logging for the daemon process.
    #
    id_system       = cfg['id_system']
    id_node         = cfg['id_node']
    level_log_event = cfg['level_log']
    if id_system and id_node:
        id_log_event = '{id_sys}.{id_node}.daemon'.format(id_sys  = id_system,
                                                          id_node = id_node)
    else:
        id_log_event = 'openai.daemon'

    (log_event, handler_log_event) = fl.log.event.logger(
                                                    str_id = id_log_event,
                                                    level  = level_log_event)

    openai.api_key = cfg['api_key']
    queue_from_api = cfg['queue_from_api']
    queue_to_api   = cfg['queue_to_api']

    log_event.info('OpenAI client is ready.')

    while True:

        # Attempt to retrieve the next request
        # from the system. If there are no
        # available requests, sleep for a short
        # period so we don't end up consuming too
        # much CPU. If a request is found, forward
        # it to OpenAI. If a response is received
        # from OpenAI, add it to the queue to be
        # sent back to the rest of the system.
        #
        try:
            request = queue_to_api.get(block = False)
        except queue.Empty:
            time.sleep(cfg['secs_interval'])
        else:
            list_msg = _process_one_request(request_raw = request,
                                            default     = cfg['default'],
                                            is_bit      = cfg['is_bit'],
                                            log_event   = log_event)
            for msg in list_msg:
                try:
                    queue_from_api.put(msg, block = False)
                except queue.Full as error:
                    log_event.exception('One or more log_metric messages ' \
                                        'dropped. queue_from_api is full.')

        # Attempt to send any available event log
        # line items to the rest of the system.
        #
        while handler_log_event.list_event:
            try:
                queue_from_api.put(
                            handler_log_event.list_event.pop(0), block = False)
            except queue.Full:
                log_event.exception('One or more log_event messages ' \
                                    'dropped. queue_from_api is full.')



# -----------------------------------------------------------------------------
def _process_one_request(request_raw, default, is_bit, log_event):
    """
    Process a single request dict, returning a result dict.

    The request_raw data structure is augmented
    with information taken from the configured
    default parameter values to produce a fully
    specified request_full data structre.

    At the same time, we determine the function
    to call to interact with the required OpenAI
    endpoint, any process state from request_raw
    to pass back to the process engine, and the
    built-in-test response to return if the bit
    flag is set.

    """
    (fcn_endpoint,
     request_full,
     state,
     response_bit) = _build_endpoint_specific_parameters(
                                                    request_raw = request_raw,
                                                    default     = default)

    result = dict(type     = 'openai_result',
                  request  = request_full,
                  response = None,
                  error    = None,
                  state    = state)
    list_msg = [result]

    if is_bit:
        result['response'] = response_bit
    else:
        log_event.info('Make request to OpenAI API.')
        try:
            result['response'] = fcn_endpoint(**request_full)
        except openai.OpenAIError as err:
            result['error'] = err.user_message
            log_event.exception('Error calling OpenAI: ' \
                                '{msg}.'.format(msg = err.user_message))
        else:

            try:
                count_token = result['response']['usage']['total_tokens']
            except KeyError as err:
                log_event.info(
                        'Response recieved from the OpenAI API. ' \
                        '(No token count).')
            else:
                log_event.info('Response recieved from the OpenAI API. ' \
                               '{num} tokens used in total.'.format(
                                                            num = count_token))
                unix_time  = request_raw.get(unix_time, 0)
                pnct       = string.punctuation
                id_model   = request_raw.get('model', '')
                id_model   = ''.join((ch for ch in id_model if ch not in pnct))
                map_state  = request_raw.get('state', dict())
                id_prompt  = map_state.get('id_prompt',  '')
                id_session = map_state.get('id_session', '')
                id_metric  = 'tokens.total'
                if id_model:
                    id_metric += '.' + id_model
                if id_prompt:
                    id_metric += '.' + id_prompt
                if id_session:
                    id_metric += '.' + id_session
                list_msg.append(dict(type    = 'log_metric',
                                     created = unix_time,
                                     id      = id_metric,
                                     value   = count_token))

    return list_msg


# -----------------------------------------------------------------------------
def _build_endpoint_specific_parameters(request_raw, default):
    """
    Return a length 4 tuple with endpoint specific parameters.

    The returned tuple contains:-

        1. The openai function to call for the requested endpoint.
        2. The request parameters fully fleshed out with default values.
        3. Process state.
        4. A built-in-test response to use when the bit flag is enabled.

    This information is extracted from the
    input request_raw structure combined with
    default values extracted from the supplied
    configuration data.

    """

    # Determine which endpoint is being requested.
    #
    id_endpoint = request_raw.get('id_endpoint', None)
    if id_endpoint is None:
        id_endpoint = default.get('id_endpoint', None)
    if id_endpoint is None:
        raise RuntimeError('Could not determine endpoint.')

    # Determine which endpoint API callback to use.
    #
    map_fcn_endpoint = dict(
        completions           = openai.Completion.create,
        chat_completions      = openai.ChatCompletion.create,
        edits                 = openai.Edit.create,
        images_generations    = openai.Image.create,
        images_edits          = openai.Image.create_edit,
        images_variations     = openai.Image.create_variation,
        embeddings            = openai.Embedding.create,
        audio_transcriptions  = openai.Audio.transcribe,
        audio_translations    = openai.Audio.translate)
    fcn_endpoint = map_fcn_endpoint[id_endpoint]

    # Determine how to handle request parameters.
    #
    # Tuple is (required-internally, required-OpenAI, name-in-api, type)
    #
    Y = True
    N = False
    map_tup_tup_param = dict(  # RI, RO, name-api,           type
        completions           = (( N,  Y, 'model',             (str,)      ),
                                 ( Y,  N, 'prompt',            (str, list) ),
                                 ( N,  N, 'suffix',            (str,)      ),
                                 ( N,  N, 'max_tokens',        (int,)      ),
                                 ( N,  N, 'temperature',       (float,)    ),
                                 ( N,  N, 'top_p',             (float,)    ),
                                 ( N,  N, 'n',                 (int,)      ),
                                 ( N,  N, 'stream',            (bool,)     ),
                                 ( N,  N, 'logprobs',          (int,)      ),
                                 ( N,  N, 'echo',              (bool,)     ),
                                 ( N,  N, 'stop',              (str, list) ),
                                 ( N,  N, 'presence_penalty',  (float,)    ),
                                 ( N,  N, 'frequency_penalty', (float,)    ),
                                 ( N,  N, 'best_of',           (int,)      ),
                                 ( N,  N, 'logit_bias',        (dict,)     ),
                                 ( N,  N, 'user',              (str,)      )),
        chat_completions      = (( N,  Y, 'model',             (str,)      ),
                                 ( Y,  Y, 'messages',          (list,)     ),
                                 ( N,  N, 'temperature',       (float,)    ),
                                 ( N,  N, 'top_p',             (float,)    ),
                                 ( N,  N, 'n',                 (int,)      ),
                                 ( N,  N, 'stream',            (bool,)     ),
                                 ( N,  N, 'stop',              (str, list) ),
                                 ( N,  N, 'max_tokens',        (int,)      ),
                                 ( N,  N, 'presence_penalty',  (float,)    ),
                                 ( N,  N, 'frequency_penalty', (float,)    ),
                                 ( N,  N, 'logit_bias',        (dict,)     ),
                                 ( N,  N, 'user',              (str,)      )),
        edits                 = (( N,  Y, 'model',             (str,)      ),
                                 ( Y,  N, 'input',             (str,)      ),
                                 ( Y,  Y, 'instruction',       (str,)      ),
                                 ( N,  N, 'n',                 (int,)      ),
                                 ( N,  N, 'temperature',       (float,)    ),
                                 ( N,  N, 'top_p',             (float,)    )),
        images_generations    = (( Y,  Y, 'prompt',            (str,)      ),
                                 ( N,  N, 'n',                 (int,)      ),
                                 ( N,  N, 'size',              (str,)      ),
                                 ( N,  N, 'response_format',   (str,)      ),
                                 ( N,  N, 'user',              (str,)      )),
        images_edits          = (( Y,  Y, 'image',             (str,)      ),
                                 ( N,  N, 'mask',              (str,)      ),
                                 ( Y,  Y, 'prompt',            (str,)      ),
                                 ( N,  N, 'n',                 (int,)      ),
                                 ( N,  N, 'size',              (str,)      ),
                                 ( N,  N, 'response_format',   (str,)      ),
                                 ( N,  N, 'user',              (str,)      )),
        images_variations     = (( Y,  Y, 'image',             (str,)      ),
                                 ( N,  N, 'n',                 (int,)      ),
                                 ( N,  N, 'size',              (str,)      ),
                                 ( N,  N, 'response_format',   (str,)      ),
                                 ( N,  N, 'user',              (str,)      )),
        embeddings            = (( N,  Y, 'model',             (str,)      ),
                                 ( Y,  Y, 'input',             (str,list)  ),
                                 ( N,  N, 'user',              (str,)      )),
        audio_transcriptions  = (( Y,  Y, 'file',              (str,)      ),
                                 ( N,  Y, 'model',             (str,)      ),
                                 ( N,  N, 'prompt',            (str,)      ),
                                 ( N,  N, 'response_format',   (str,)      ),
                                 ( N,  N, 'temperature',       (float,)    ),
                                 ( N,  N, 'language',          (str,)      )),
        audio_translations    = (( Y,  Y, 'file',              (str,)      ),
                                 ( N,  Y, 'model',             (str,)      ),
                                 ( N,  N, 'prompt',            (str,)      ),
                                 ( N,  N, 'response_format',   (str,)      ),
                                 ( N,  N, 'temperature',       (float,)    )))

    # Build the fully fleshed out request_full
    # data structure.
    #
    request_full    = dict()
    tup_tup_param = map_tup_tup_param[id_endpoint]
    for (is_required_by_internal_api,  # Cannot be given a default value.
         is_required_by_openai_api,    # Must be given a value of some sort.
         id_param,
         tup_type) in tup_tup_param:

        # Try to get value from request_raw.
        #
        param = request_raw.get(id_param, None)
        if param is not None:
            _check_type(id_param, param, tup_type)
            request_full[id_param] = param
            continue

        if is_required_by_internal_api:
            msg = '{id} not in request_raw'.format(id = id_param)
            log_event.error(msg)
            raise RuntimeError(msg)

        # Try to get value from default.
        #
        param = default.get(id_param, None)
        if param is not None:
            _check_type(id_param, param, tup_type)
            request_full[id_param] = param
            continue

        if is_required_by_openai_api:
            msg = '{id} not in request_raw or default'.format(id = id_param)
            log_event.error(msg)
            raise RuntimeError(msg)

    state = request_raw.get('state', {})

    # Work out the response to give when the
    # built-in-test flag is enabled.
    #
    response_bit = built_in_test_response(id_endpoint)

    return (fcn_endpoint, request_full, state, response_bit)


# -----------------------------------------------------------------------------
def _check_type(id_param, param, tup_type):
    """
    Raise an exception if param is not in the specified tup_type.

    """

    if not isinstance(param, tup_type):
        msg = 'Type error for request["{id}"]: ' \
              '({typename} != {required})'.format(id       = id_param,
                                                  typename = type(param),
                                                  required = repr(tup_type))
        log_event.error(msg)
        raise RuntimeError(msg)


# -----------------------------------------------------------------------------
def built_in_test_response(id_endpoint, id_version = 'v1'):
    """
    Return the built-in-test response for the specified endpoint.

    This function takes an id_endpoint and an
    optional id_version (default 'v1') and returns
    a built-in-test response for the specified
    endpoint.

    The built-in-test responses cover a variety
    of API endpoints such as completions,
    chat_completions, edits, images_generations,
    images_edits, images_variations, embeddings,
    audio_transcriptions, and audio_translations.

    Args:

    id_endpoint (str):          The identifier of the desired API endpoint,
                                e.g. 'completions', 'chat_completions', etc.
    id_version (str, optional): The version of the API endpoint.
                                Defaults to 'v1'.

    Returns:

        dict:                   A dictionary containing the built-in-test
                                response for the specified endpoint.

    Examples:

    >>> built_in_test_response('completions')
    {'id': 'cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7', 'object': 'text_completion', ...}

    """
    map_bit_endpoint = {
        ('v1', 'completions'): {
            'id':       'cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7',
            'object':   'text_completion',
            'created':  1589478378,
            'model':    'text-davinci-003',
            'choices':  [ { 'text':                 '\n\nThis is a test',
                            'index':                0,
                            'logprobs':             None,
                            'finish_reason':        'length'}],
            'usage':    {   'prompt_tokens':        10,
                            'completion_tokens':    20,
                            'total_tokens':         30 }},

        ('v1', 'chat_completions'): {
            'id':       'chatcmpl-abc123',
            'object':   'chat.completion',
            'created':  1677858242,
            'model':    'gpt-3.5-turbo-0301',
            'choices':  [ { 'message':              { 'role':    'assistant',
                                                      'content': '\n\nTest'},
                            'finish_reason':        'stop',
                            'index':                0}],
            'usage':    {   'prompt_tokens':        10,
                            'completion_tokens':    20,
                            'total_tokens':         30}},

        ('v1', 'edits'): {
            'object':   'edit',
            'created':  1589478378,
            'choices':  [ { 'text':                 'Test edit response',
                            'index':                0}],
            'usage':    {   'prompt_tokens':        10,
                            'completion_tokens':    20,
                            'total_tokens':         30}},

        ('v1', 'images_generations'): {
            'created': 1589478378,
            'data':     [ { 'url': 'https://random_test_url_424242.co.uk' },
                          { 'url': 'https://random_test_url_424242.co.uk' }]},

        ('v1', 'images_edits'): {
            'created': 1589478378,
            'data':     [ { 'url': 'https://random_test_url_424242.co.uk' },
                          { 'url': 'https://random_test_url_424242.co.uk' }]},

        ('v1', 'images_variations'): {
            'created': 1589478378,
            'data':     [ { 'url': 'https://random_test_url_424242.co.uk' },
                          { 'url': 'https://random_test_url_424242.co.uk' }]},

        ('v1', 'embeddings'): {
            'object':   'list',
            'model':    'text-embedding-ada-002',
            'data':     [ { 'object':               'embedding',
                            'embedding':            [ 0.0023064255,
                                                     -0.009327292,
                                                     -0.0028842222],
                            'index':                0}],
            'usage':    {   'prompt_tokens':        10,
                            'total_tokens':         20}},

        ('v1', 'audio_transcriptions'): {
            'text':     'Test transcription response.' },

        ('v1', 'audio_translations'): {
            'text':     'Test translation response' }}

    return map_bit_endpoint[(id_version, id_endpoint)]
