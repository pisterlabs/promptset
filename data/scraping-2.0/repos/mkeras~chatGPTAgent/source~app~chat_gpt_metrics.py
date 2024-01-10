from app import openai_functions, env, logging
from typing import List
from functools import partial
from sparkplug_node_app.sparkplug_tags import SparkplugDataTypes, SparkplugMemoryTag

def debug(message: str):
    logging.debug(f'chat_gpt_options module: {message}')

CREATE_CHAT_PARAMETERS = {
    'model': {
        'datatype': SparkplugDataTypes.String,
        'default': env.GPT_MODEL,
        'always_required': True,
        'validate': {
            'options': openai_functions.list_gpt_models()
        }
    },
    'frequency_penalty': {
        'datatype': SparkplugDataTypes.Float,
        'default': 0.0,
        'validate': {
            'min': -2.0,
            'max': 2.0
        }
    },
    'presence_penalty': {
        'datatype': SparkplugDataTypes.Float,
        'default': 0.0,
        'validate': {
            'min': -2.0,
            'max': 2.0
        }
    },
    'max_tokens': {
        'datatype': SparkplugDataTypes.Int64,
        'default': None,
        'validate': {
            'min': 100,
        }
    },
    'temperature': {
        'datatype': SparkplugDataTypes.Float,
        'default': 1.0,
        'validate': {
            'min': 0.0,
            'max': 2.0
        }
    },
    'response as json': {
        'datatype': SparkplugDataTypes.Boolean,
        'default': False
    }
}

PREFIX = 'Agent Config/chatGPT Parameters/'

def _tag_name(name: str, prefix: str = PREFIX) -> str:
    return f'{prefix}{name}'

def _validate_options(current_value, new_value, options: List[str]) -> bool:
    valid = new_value in options
    if valid:
        logging.debug(f'"{new_value}" is valid option!')
    else:
        logging.warning(f'"{new_value}" is not a valid option!')
    return valid

def _validate_range(current_value, new_value, min_: int or float = None, max_: int or float = None) -> bool:
    if new_value is None:
        logging.warning(f'new_value is None, assuming valid!')
        return False
    if min_ is None and max_ is None:
        logging.warning(f'No min or max is supplied!')
        return True
    if min_ is not None and new_value < min_:
        logging.warning(f'Value "{new_value}" is below the minimum value of "{min_}"')
        return False
    if max_ is not None and new_value > max_:
        logging.warning(f'Value "{new_value}" is above the maximum value of "{max_}"')
        return False
    return True

chat_metrics = []

for parameter, metadata in CREATE_CHAT_PARAMETERS.items():
    create_tag_args = dict(
        name=_tag_name(parameter),
        writable=True,
        disable_alias=True,
        persistence_file=env.MEMORY_TAGS_FILEPATH
    )

    if 'datatype' in metadata.keys():
        create_tag_args['datatype'] = metadata['datatype']
    elif 'datatype_code' in metadata.keys():
        create_tag_args['datatype'] = SparkplugDataTypes(metadata['datatype_code'])
    else:
        raise ValueError(f'No datatype given for metric "{create_tag_args["name"]}"')

    if 'default' in metadata.keys():
        create_tag_args['initial_value'] = metadata['default']

    validate = metadata.get('validate')
    if not validate:
        pass
    elif 'options' in validate.keys():
        create_tag_args['write_validator'] = partial(_validate_options, options=validate['options'])
    elif 'max' in validate.keys() or 'min' in validate.keys():
        create_tag_args['write_validator'] = partial(_validate_range, min_=validate.get('min'), max_=validate.get('max'))

    chat_metrics.append(SparkplugMemoryTag(**create_tag_args))


def get_create_chat_args(metrics: List[SparkplugMemoryTag] = chat_metrics, parameters: dict = CREATE_CHAT_PARAMETERS, read_metrics: bool = False) -> dict:
    args = {}
    for metric in metrics:
        if read_metrics:
            metric.read()
        parameter_key = metric.name.split('/')[-1]  # get the actual name without path
        always_required = parameters[parameter_key].get('always_required')
        if metric.current_value is None and not always_required:
            logging.debug(f'SKIPPING "{metric.name}", current value is None')
            continue

        if 'default' in parameters[parameter_key].keys() and parameters[parameter_key]['default'] == metric.current_value and not always_required:
            logging.debug(f'SKIPPING "{metric.name}" default value')
            continue

        if parameter_key == 'response as json' and metric.current_value:
            args['response_format'] = {'type': 'json_object'}
            continue

        args[parameter_key] = metric.current_value

    return args


def reset_to_defaults(metrics: List[SparkplugMemoryTag] = chat_metrics, parameters: dict = CREATE_CHAT_PARAMETERS):
    for metric in metrics:
        parameter_key = metric.name.split('/')[-1]  # get the actual name without path
        if 'default' not in parameters[parameter_key].keys():
            continue
        metric.write(parameters[parameter_key].get('default'))
        logging.debug(f'Reset "{metric.name}" to default value')
