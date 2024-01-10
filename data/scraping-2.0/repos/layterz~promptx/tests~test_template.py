import pytest
import openai
import json

from promptx.template import *
from promptx.models import Response

from . import User, Trait, session, llm


@pytest.fixture
def template(session):
    t = Template(instructions='Some example instructions', output=json.dumps(User.model_json_schema()))
    session.store(t, collection='templates')
    return session.query(ids=[t.id], collection='templates').first


def test_basic_response(session, llm):
    template = Template(instructions='Some example instructions')
    o = session.prompt(template=template, llm=llm)
    assert o is not None
    assert o == 'This is a mock response.'

def test_json_valid_output(session, template, llm):
    llm.generate.return_value = Response(
        raw='{ "name": "test", "age": 20, "traits": ["nice"] }',
    )
    o = session.prompt(template=template, llm=llm)

    assert o is not None
    assert o.type == 'user'
    assert o.name == 'test'
    assert o.age == 20

def test_load_from_template_id(session, template, llm):
    llm.generate.return_value = Response(
        raw='{ "name": "test", "age": 20, "traits": ["nice"] }',
    )
    o = session.prompt(template=template.id, llm=llm)
    assert o is not None
    assert o.type == 'user'
    assert o.name == 'test'
    assert o.age == 20
    
def test_json_valid_output__extra_field(session, template, llm):
    response = Response(
        raw='{ "name": "test", "age": 20, "location": "london", "traits": ["nice"] }',
    )
    llm.generate.return_value = response

    o = session.prompt(template=template, llm=llm)
    assert o.name == 'test'
    assert o.age == 20
    with pytest.raises(AttributeError):
        assert o.location == 'london'
    
def test_json_invalid_output__missing_required_field(session, template, llm):
    response = Response(
        raw='{ "age": 20 }',
    )
    llm.generate.return_value = response

    with pytest.raises(MaxRetriesExceeded):
        session.prompt(template=template, llm=llm)
    
def test_json_invalid_output__formatting(session, template, llm):
    response = Response(
        raw='"name": "test", "age": 20, "traits": ["nice"] }',
    )
    llm.generate.return_value = response

    with pytest.raises(MaxRetriesExceeded):
        session.prompt(template=template, llm=llm)
    
def test_invalild_json_output__validation(session, template, llm):
    response = Response(
        raw='{ "name": "test", "age": "young" }',
    )
    llm.generate.return_value = response

    with pytest.raises(MaxRetriesExceeded):
        session.prompt(template=template, llm=llm)

# TODO: this should probably have some kind of separate retry budget
def test_exception_handling(session, template, llm):
    llm.generate.side_effect = [openai.Timeout, Response(raw='Test response')]
    template = Template(instructions='Some example instructions')
    
    o = session.prompt(template=template, llm=llm)
    assert o == 'Test response'

def test_parse_exception_handling(session, mocker, template, llm):
    mocker.patch.object(TemplateRunner, 'process', side_effect=[*[json.JSONDecodeError('test', 'test', 0)] * 4, 'test'])
    runner = TemplateRunner()

    with pytest.raises(MaxRetriesExceeded):
        o = runner(session, template, None, llm=llm)
    
    mocker.patch.object(TemplateRunner, 'process', side_effect=[*[json.JSONDecodeError('test', 'test', 0)] * 3, 'test'])
    runner = TemplateRunner()
    o = runner(session, template, None, llm=llm)
    
    assert o.content == 'test'

def test_invalid_input_raises_error(session, template, llm):
    with pytest.raises(MaxRetriesExceeded):
        session.prompt(template=template, input={'age': 'young'}, llm=llm)

def test_output_parsing(session, template, llm):
    llm.generate.return_value = Response(raw='{ "name": "test", "age": 20, "traits": ["nice"] }')

    o = session.prompt(template=template, llm=llm)
    assert o.type == 'user'
    assert o.name == 'test'
    assert o.age == 20

def test_format_rendering(template):
    runner = TemplateRunner()
    p = runner.render(template, {})
    assert template.instructions in p

def test_format_rendering_with_input(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'Some test input' in p

def test_format_rendering_with_output(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'name (type: string, required: True, default: None' in p

def test_format_rendering_object(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'Return the output as a valid JSON object with the fields described below' in p

def test_format_rendering_list():
    schema = json.dumps({
        'type': 'array',
        'items': {}
    })
    t = Template(instructions='Some example instructions', output=schema)
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'Return a list of valid JSON objects with the fields described below' in p

def test_format_rendering_with_basic_types(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'name (type: string, required: True, default: None' in p
    assert 'age (type: integer, required: True, default: None' in p

def test_format_rendering_with_enum(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'role (type: string, required: False, default: admin' in p
    assert 'Select one option from: admin, user' in p

def test_format_rendering_with_enum_list(session, template, llm):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'traits (type: string[], required: True, default: None' in p
    assert 'Select any relevant options from: nice, mean, funny, smart' in p

def test_format_rendering_with_excluded_fields(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'banned (type: bool, required: False, default: False' not in p

@pytest.mark.skip(reason="Not implemented yet")
def test_format_rendering_with_field_description(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'What kind of personality describes the user?' in p

def test_format_rendering_with_field_min_max(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'minimum: 18' in p
    assert 'exclusiveMaximum: 100' in p

def test_format_rendering_with_field_min_max_items(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'minItems: 1' in p
    assert 'maxItems: 3' in p

def test_format_rendering_with_field_min_max_length(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'minLength: 3' in p
    assert 'maxLength: 20' in p

def test_example_rendering(session):
    user = User(name="John Wayne", age=64, traits=[Trait.mean])
    runner = TemplateRunner()
    template = Template(instructions='Some example instructions', output=user.model_dump_json(),
                        examples=[Example(input='Some test input', output=user.model_dump_json())])
    session.store(template)
    template_r = session.query(ids=[template.id]).first
    p = runner.render(template_r, {'input': 'Some test input'})

    assert 'EXAMPLES' in p
    assert 'John Wayne' in p
    assert '64' in p
    assert 'mean' in p
    assert 'banned' in p

def test_example_rendering_multiple(session, template):
    user = User(name="John Wayne", age=64, traits=[Trait.mean])
    runner = TemplateRunner()
    examples = [Example(input='Some test input', output=user.model_dump_json()) for _ in range(3)]
    template = Template(instructions='Some example instructions', output=user.model_dump_json(), examples=examples)
    session.store(template)
    template = session.query(ids=[template.id]).first
    p = runner.render(template, {'input': 'Some test input'})

    assert p.count('John Wayne') == 3