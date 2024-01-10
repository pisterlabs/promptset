""" doc """
import functools
import json

from dotenv import load_dotenv
from jinja2 import Template, meta, Environment

from func_ai.utils.jinja_template_functions import JinjaOpenAITemplateFunction
from func_ai.utils.llm_tools import OpenAIInterface
from func_ai.utils.py_function_parser import func_to_json


def render_template(template_file: str, **params) -> str:
    """
    Create subscription in the network using a template and given parameters by the user.

    :param template_file:
    :param params: parameters to be used in the template
    :return:
    """
    # render the jinja template with the parameters

    with open(template_file) as f:
        prompt = f.read()
        print(Template(prompt).render(**params))
    return "ok"


def t1(a: str):
    """
    wqewqeq

    :param a:
    :return:
    """


def test_template_fun():
    load_dotenv()
    inf = OpenAIInterface()
    inf.conversation_store.add_system_message("""You are a code helper. Your goal is to help the user to convert a jinja template into a list of parameters.

User will provide a jinja template as input.

Your task is to extract the jinja parameters and return them in a bullet list.

Do not return anything other than the bullet list.
Return just the parameter names and nothing else.
Return only jinja2 template parameters and nothing else.
""")
    with open("template2.txt") as f:
        prompt = f.read()
        resp = inf.send(prompt=prompt)
        dynamic_args = resp['content'].replace("-", "").split("\n")
        # here we want to use partial so that we can se the template file
        _fun_partial = functools.partial(render_template, template_file="template2.txt")
        _fun = func_to_json(_fun_partial)
        _fun["parameters"]['properties'] = {k: {"type": "string", "description": f"{k}"} for k in dynamic_args}
        _fun['required'] = [k for k in dynamic_args]
        _t_prompt = "I am John. I live in France"
        _fun["description"] = f"Render Template\nAccepts the following {resp['content']}"
        inf.conversation_store.add_system_message(
            """Help the user run a template with the parameters provided by the user.""")
        resp2 = inf.send(prompt=_t_prompt,
                         functions=[_fun])
        print(f"LLM Response: {resp2}")
        if "function_call" in resp2:
            args = json.loads(resp2["function_call"]["arguments"])
            print(_fun_partial(**args))


def test_template_xml():
    load_dotenv()
    inf = OpenAIInterface()
    inf.conversation_store.add_system_message("""You are a code helper. Your goal is to help the user to convert a jinja template into a list of parameters.

User will provide a jinja template as input.

Your task is to extract the jinja parameters and return them in a bullet list.

Do not return anything other than the bullet list.
Return just the parameter names and nothing else.
Return only jinja2 template parameters and nothing else.
""")
    with open("template.xml") as f:
        prompt = f.read()
        resp = inf.send(prompt=prompt)
        dynamic_args = resp['content'].replace("-", "").split("\n")
        # here we want to use partial so that we can se the template file
        _fun_partial = functools.partial(render_template, template_file="template.xml")
        _fun = func_to_json(_fun_partial)
        _fun["parameters"]['properties'] = {k: {"type": "string", "description": f"{k}"} for k in dynamic_args}
        _fun['required'] = [k for k in dynamic_args]
        _t_prompt = "Create user 1 John"
        _fun["description"] = f"Render Template\nAccepts the following {resp['content']}"
        inf.conversation_store.add_system_message(
            """Help the user run a template with the parameters provided by the user.""")
        resp2 = inf.send(prompt=_t_prompt,
                         functions=[_fun])
        print(f"LLM Response: {resp2}")
        if "function_call" in resp2:
            args = json.loads(resp2["function_call"]["arguments"])
            print(_fun_partial(**args))
            print(f"Cost: {inf.get_usage()}")


def test_template_vars():
    _t = Template("{{ a }}")
    env = Environment()
    ast = env.parse("{{ a }}")
    # source = _t.environment.loader.get_source(_t.environment, _t.name)

    print(meta.find_undeclared_variables(ast))  # prints: {'a'}


def test_jinja_template_object():
    load_dotenv()
    ji = JinjaOpenAITemplateFunction.from_string_template("Name: {{ NAME }} \n Age: {{ AGE }}", OpenAIInterface())
    resp = ji.render_from_prompt("John is 20 years old")
    assert "Name: John" in resp
    assert "Age: 20" in resp
    # prints
    """
    Name: John 
    Age: 20
    """


def test_jinja_template_object_complex_sentence():
    load_dotenv()
    ji = JinjaOpenAITemplateFunction.from_string_template("Name: {{ NAME }} \n Age: {{ AGE }}", OpenAIInterface())
    resp = ji.render_from_prompt("His name is John, He's born in Sofia and he's born 1983. We're in 2023.")
    print(resp)
    assert "Name: John" in resp
    assert "Age: 40" in resp
    # prints
    """
    Name: John 
    Age: 20
    """


def test_jinja_template_object_with_jinja_function():
    load_dotenv()
    ji = JinjaOpenAITemplateFunction.from_string_template("Name: {{ NAME |upper}} \n Age: {{ AGE }}", OpenAIInterface())
    resp = ji.render_from_prompt("John is 20 years old")
    assert "Name: JOHN" in resp
    assert "Age: 20" in resp
    print(resp)
    # prints
    """
    Name: JOHN 
    Age: 20
    """


def assert_is_digit(x):
    assert x.isdigit()
    return x


def test_jinja_template_object_with_filters_function():
    load_dotenv()
    ji = JinjaOpenAITemplateFunction.from_string_template("Name: {{ NAME }} \n Age: {{ AGE |validate_digit}}",
                                                          OpenAIInterface(),
                                                          filters={"validate_digit": assert_is_digit})
    resp = ji.render_from_prompt("John is 20y")
    print(resp)
    assert "Name: John" in resp
    assert "Age: 20" in resp
    # prints
    """
    Name: John 
    Age: 20
    """


def test_single_prompt():
    load_dotenv()
    llm = OpenAIInterface()
    _resp = llm.send(
        prompt="Create a user with name John and age 20\n Render as `Name: {{ NAME }} \n Age: {{ AGE }}`")

    print(_resp['content'])
