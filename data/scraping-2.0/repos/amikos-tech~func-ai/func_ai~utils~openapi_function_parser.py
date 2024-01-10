""" doc """
import functools
import os

import requests
import json

from func_ai.function_indexer import FunctionIndexer
from func_ai.utils.llm_tools import OpenAIInterface, OpenAIFunctionWrapper
from func_ai.utils.py_function_parser import func_to_json

curdir = os.path.dirname(os.path.abspath(__file__))


def get_spec_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception in case of a failure
    return response.json()


def _coerce_type(value: any, value_type: type) -> any:
    """
    Coerce the value to the given type. If the type is not supported, return the value as is.

    Note: this is not great, not even good but it is a start.

    :param value: The value to be coerced
    :param value_type: The type to coerce to
    :return:
    """
    if value_type == "integer":
        return int(value)
    if value_type == "number":
        return float(value)
    if value_type == "boolean":
        return bool(value)
    return value


def get_operations_from_path_item(path_item):
    """
    Get the operations from a path item

    :param path_item:
    :return:
    """
    http_methods = ["get", "put", "post", "delete", "options", "head", "patch", "trace"]
    operations = [{"method": op, "spec": op_spec} for op, op_spec in path_item.items() if op in http_methods]
    return operations


def get_operation_details(operation, path):
    """
    Get the details of an operation

    :param operation:
    :param path:
    :return:
    """
    return {
        "name": operation['spec'].get('operationId'),
        "description": operation['spec'].get('description'),
        "parameters": operation['spec'].get('parameters'),
        "summary": operation['spec'].get('summary', ''),
        "method": operation['method'],
        "consumes": operation['spec'].get('consumes', []),
        "produces": operation['spec'].get('produces', []),
        "responses": operation['spec'].get('responses', {}),
        "security": operation['spec'].get('security', []),
        "tags": operation['spec'].get('tags', []),
        "path": path,
    }


def parse_parameters(parameters, _defs):
    """
    Parse the parameters of an operation and return a dict that can be used to generate a function signature.

    :param parameters:
    :param _defs:
    :return:
    """
    params_dict = {"type": "object", "properties": {}, "required": []}

    for param in parameters:
        name = param['name']
        description = param.get('description', '')  # TODO generate a description if not present
        if 'schema' in param and '$ref' in param['schema']:
            _param_model = param.get('schema').get('$ref').replace('#/definitions/', '')
            param_model = _defs.get(_param_model)
            for p, pd in param_model['properties'].items():
                if '$ref' in pd:
                    param_model['properties'][p] = _defs.get(pd['$ref'].replace('#/definitions/', ''))
                if 'items' in pd and '$ref' in pd['items']:
                    param_model['properties'][p]['items'] = _defs.get(pd['items']['$ref'].replace('#/definitions/', ''))
            description += ".\nJSON Model: " + json.dumps(param_model)
        param_type = param['schema'].get('type', 'string') if 'schema' in param else 'string'
        # TODO if parameter is schema then we need to go down the rabbit hole and get the schema
        params_dict["properties"][name] = {
            "description": description,
            "type": param_type,
            "in": param['in'] if 'in' in param else "query"
        }

        if param.get('required', False):
            params_dict["required"].append(name)

    return params_dict


def get_func_details(operation, _defs):
    """
    Generate the function details from the operation details

    :param operation:
    :param _defs:
    :return:
    """
    name = operation.get('name')
    description = operation.get('summary', '') + operation.get(
        'description')  # TODO if description not present generate one with AI
    parameters = operation.get('parameters', [])

    # Parse the parameters
    parameters = parse_parameters(parameters, _defs)

    return {
        "name": name,
        "description": description,
        "parameters": parameters
    }


def parse_spec(spec) -> dict[str, dict]:
    """
    Parse the OpenAPI specification and return a dictionary of functions that can be called

    :param spec:
    :return:
    """
    _defs = spec.get('definitions')
    paths = spec['paths']
    _funcs = {}
    for path, path_item in paths.items():
        operations = get_operations_from_path_item(path_item)
        for operation in operations:
            details = get_operation_details(operation, path)
            _funcs[details['name']] = {"details": details, "func": get_func_details(details, _defs),
                                       "operation_raw": operation}
    return _funcs


def _api_call(action: OpenAIFunctionWrapper, **kwargs):
    """
    Call the API

    :param action:
    :param kwargs:
    :return:
    """
    _f_name = action.name
    _body_params = action.metadata["body_params"]
    _query_params = action.metadata["query_params"]
    _path_params = action.metadata["path_params"]
    _path = action.metadata["path"]
    _method = action.metadata["method"]
    _url_spec = action.metadata["url_spec"]["template_url"].format(**{**action.metadata["url_spec"], "path": _path})
    if len(_path_params):
        _url_spec = _url_spec.format(
            **{param['name']: _coerce_type(kwargs.get(param['name']), param['type']) for param in _path_params})
    if len(_query_params):
        _url_spec += "?" + "&".join(
            [f"{param['name']}={kwargs.get(param['name'])}" for param in _query_params])
    _body_param_name = None if len(_body_params) == 0 else _body_params[0]['name']
    api_resp = None
    # TODO headers should comply with the spec
    if _method == "post":
        api_resp = requests.post(_url_spec, json=json.loads(kwargs.get(_body_param_name)),
                                 headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
    elif _method == "put":
        api_resp = requests.put(_url_spec, json=json.loads(kwargs.get(_body_param_name)),
                                headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
    elif _method == "patch":
        api_resp = requests.patch(_url_spec, json=json.loads(kwargs.get(_body_param_name)),
                                  headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
    elif _method == "get":
        api_resp = requests.get(_url_spec,
                                headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
    else:
        raise ValueError(f"Method {_method} not supported")
    return {
        "status_code": api_resp.status_code,
        "response": api_resp.text
    }


def _read_system_prompt_file(path: str) -> str:
    """
    Read the system prompt file

    :param path:
    :return:
    """
    with open(path, "r") as f:
        return f.read()


## QA Utility Functions
def list_available_functions(spec_wrapper: "OpenAPISpecOpenAIWrapper", **kwargs) -> str:
    """
    List the available API operations
a
    :param spec_wrapper: The OpenAPI Spec Wrapper
    :return:
    """
    return "Available functions:\n" + "\n".join([f"- {k}: {v}" for k, v in spec_wrapper.operations_summary.items()])


def get_operation_function_info(operation_name: str, spec_wrapper: "OpenAPISpecOpenAIWrapper", **kwargs) -> dict[
    str, any]:
    """a
    Returns the full information about an operation including all parameters

    :param operation_name: The name of the operation
    :param spec_wrapper: The OpenAPI Spec Wrapper
    :return:
    """

    return spec_wrapper.get_operation(operation_name).to_dict()


class OpenAPISpecOpenAIWrapper(object):
    """
    Class that wrap around the OpenAPI specification and provides a set of methods to interact with it using OpenAI
    function calling.
    """

    def __init__(self, spec: dict[any, any],
                 llm_interface: OpenAIInterface,
                 source=None,
                 index: bool = False, **kwargs) -> None:
        """
        Initialize the OpenAPI wrapper
        Note: When index=True the LLM system message will be overwritten with the system message from the spec QA system prompt file

        :param spec: The OpenAPI specification
        :param llm_interface: The LLM Interface
        :param source: The source of the spec
        :param index: Whether to index the spec (default: False) - this will put all operations in a vector store and make them searchable
        :param kwargs: Additional arguments
        """
        self.spec = spec
        self.llm_interface = llm_interface
        self.source = source
        _funcs = parse_spec(spec)
        self._function_calls = []
        self._url_spec = {
            "host": spec["host"],
            "base_path": spec["basePath"],
            "schemes": spec["schemes"],
            "default_scheme": "https" if "https" in spec["schemes"] else "http",
            "template_url": "{default_scheme}://{host}{base_path}{path}",
        }
        self._funcs = {
            fn: OpenAIFunctionWrapper(llm_interface=self.llm_interface,
                                      name=fn,
                                      description=f['func']['description'],
                                      parameters=f['func']['parameters'],
                                      func=functools.partial(
                                          _api_call, ),
                                      # metadata
                                      **{
                                          "url_spec": self._url_spec,
                                          "body_params": [param for param in f['details']['parameters']
                                                          if
                                                          param['in'] == 'body'],
                                          "path_params": [param for param in f['details']['parameters']
                                                          if
                                                          param['in'] == 'path'],
                                          "query_params": [param for param in f['details']['parameters']
                                                           if
                                                           param['in'] == 'query'],
                                          "method": f['details']['method'],
                                          "path": f['details']['path'],
                                          **kwargs
                                      }) for
            fn, f in _funcs.items()}
        if index:
            self.indexer = FunctionIndexer(llm_interface=self.llm_interface)
            # TODO: here we're resetting the index every time we create a new wrapper. In the future we should be able to index the same spec multiple times
            self.indexer.reset_function_index()
            self.indexer.index_wrapper_functions([f for k, f in self.operations.items()])
            self.llm_interface.conversation_store.add_system_message(
                _read_system_prompt_file(os.path.join(curdir, 'api_qa_system_prompt.txt')))

    @property
    def operations(self) -> dict[str, any]:
        """
        Return the dictionary of functions that can be called

        :return:
        """
        return self._funcs

    @property
    def operations_summary(self) -> dict[str, str]:
        """
        Return a summary of the operations in the form of a dictionary
        where the key is the operation name and the value is the description

        :return: A dictionary of operations
        """
        return {fn: f.description for fn, f in self._funcs.items()}

    def to_dict(self) -> dict[str, any]:
        """
        Return the dictionary of functions that can be called

        :return:
        """
        return {fn: f.to_dict() for fn, f in self._funcs.items()}

    def get_operation(self, name: str) -> OpenAIFunctionWrapper:
        """
        Get a function by name

        :param name: The name of the function
        :return:
        """
        return self[name]

    @classmethod
    def from_url(cls, url: str, llm_interface: OpenAIInterface, **kwargs) -> "OpenAPISpecOpenAIWrapper":
        """
        Create an instance of the class from a URL

        :param url: The URL to load
        :param llm_interface: The interface to use
        :param kwargs: Additional arguments
        :return:
        """
        spec = get_spec_from_url(url)
        return cls(spec, llm_interface=llm_interface, source={"url": url, "type": "url"}, **kwargs)

    @classmethod
    def from_file(cls, file: str, llm_interface: OpenAIInterface, **kwargs) -> "OpenAPISpecOpenAIWrapper":
        """
        Create an instance of the class from a file

        :param file: The file to load
        :param llm_interface: The interface to use
        :return:
        """
        with open(file) as f:
            spec = json.load(f)

        return cls(spec, llm_interface=llm_interface, source={"url": file, "type": "file"}, **kwargs)

    def __getitem__(self, item) -> OpenAIFunctionWrapper:
        """
        Get a function from the wrapper
        :param item: The name of the function
        :return: The function wrapper
        """
        return self._funcs[item]

    def from_prompt(self, prompt: str, operation_name: str, **kwargs) -> "OpenAPISpecOpenAIWrapper":
        """
        Prompt the user for a function to call and the parameters to use
        :param prompt: The prompt to use
        :param operation_name: The name of the operation to use
        :param kwargs: Additional arguments to pass to the prompt
        :return: The result of the function call
        """
        _fwrap = self._funcs[operation_name].from_prompt(prompt=prompt, **kwargs)
        self._function_calls.append(_fwrap.last_call)
        return self

    @property
    def last_call(self) -> dict[str, any]:
        """
        Return the last function call
        :return:
        """
        return self._function_calls[-1]

    @property
    def calls(self) -> list[dict[str, any]]:
        """
        Return the list of function calls
        :return:
        """
        return self._function_calls

    def api_qa(self, prompt: str, **kwargs) -> any:
        """
        Ask a question about the API
        :param prompt: The prompt to use
        :param kwargs: Additional arguments to pass to the prompt
        :return: The result of the function call
        """
        # What operations does this API support?
        _list_fun = OpenAIFunctionWrapper.from_python_function(
            func=functools.partial(list_available_functions, spec_wrapper=self),
            llm_interface=self.llm_interface, )
        # What endpoints are available in this API?

        # What are the mandatory parameters for operation X?
        _fun_info = OpenAIFunctionWrapper.from_python_function(
            func=functools.partial(get_operation_function_info, spec_wrapper=self),
            llm_interface=self.llm_interface, )
        # What are the optional parameters for operation X?

        _f_map = {_list_fun.name: _list_fun,
                  _fun_info.name: _fun_info, }
        _llm_resp = self.llm_interface.send(prompt=prompt, functions=[k.schema for _, k in _f_map.items()], **kwargs)
        # print(f"LLM Response: {_llm_resp}")
        if "function_call" in _llm_resp:
            _fcall = _f_map[_llm_resp["function_call"]["name"]].from_response(llm_response=_llm_resp).last_call[
                'function_response']
            # print(f"Function call: {_fcall}")
            self.llm_interface.add_conversation_message(_fcall, update_llm=True, **kwargs)
            return self.llm_interface.conversation_store.get_last_message()['content']
            # _llm_secondary_resp = self.llm_interface.send(prompt=prompt,
            #                                               functions=[k.schema for _, k in _f_map.items()],
            #                                               **kwargs)
        else:
            return _llm_resp['content']
