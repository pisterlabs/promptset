import time
import random
import json
from typing import * 
import jsonschema
from loguru import logger
from pydantic import BaseModel
import openai
from jinja2 import Template as JinjaTemplate

from .collection import Collection, Entity, Query, model_to_json_schema, create_entity_from_schema
from .models import MockLLM


class MaxRetriesExceeded(Exception):
    pass


E = TypeVar('E', bound=BaseModel)


class Example(Entity):
    input: str
    output: str

    def __init__(self, input, output, **kwargs):
        super().__init__(input=self.parse(input), output=self.parse(output), **kwargs)
    
    def parse(self, x):
        def _serialize(x):
            if issubclass(type(x), BaseModel):
                return x.model_dump()
        if isinstance(x, str):
            return x
        else:
            return json.dumps(x, default=_serialize)


class Template(Entity):
    
    template: str = """
    INSTRUCTIONS
    ---
    {{instructions}}
    {{format}}
    {% if examples %}
        EXAMPLES
        ---
    {% endif %}
    {{examples}}
    {% if examples %}
        END_EXAMPLES
        ---
    {% endif %}
    {{input}}
    {{output}}
    """

    input_template: str = """
    INPUT
    ---
    {{input}}
    END_INPUT
    """

    output_template: str = """
    OUTPUT
    ---
    {{output}}
    """

    example_template: str = f"""
    {input_template}
    {output_template}
    """

    format_template: str = """
    FORMAT INSTRUCTIONS
    ---
    {% if string_list_output %}
    Return a JSON array of strings.
    {% elif list_output %}
    Return a list of valid JSON objects with the fields described below.
    {% else %}
    Return the output as a valid JSON object with the fields described below. 
    {% endif %}
    {% for field in fields %}
    - {{field.name}} (type: {{field.type_}}, required: {{field.required}}, default: {{field.default}}, {% for k, v in field.metadata.items()%}{{k}}: {{v}}, {% endfor %}): {{field.description}}
    {% endfor %}

    Make sure to use double quotes and avoid trailing commas!
    Ensure any required fields are set, but you can use the default value 
    if it's defined and you are unsure what to use. 
    If you are unsure about any optional fields use `null` or the default value,
    but try your best to fill them out.
    END_FORMAT_INSTRUCTIONS
    """

    type: str = 'template'
    name: str = None
    instructions: str = None
    examples: List[Example] = None
    input: str = None
    output: str = None
    context: str = None
    data: Query = None

    def __init__(self, examples=None, **kwargs):
        if examples is not None:
            for i, example in enumerate(examples):
                if isinstance(example, dict):
                    example = Example(**example)
                elif isinstance(example, tuple):
                    example = Example(*example)
                if not isinstance(example, Example):
                    continue
                examples[i] = example
            kwargs['examples'] = examples
        super().__init__(**kwargs)


class TemplateRunner:
    
    def parse(self, x):
        if x is None:
            return {}
        skip_list = ['id', 'type']
        if isinstance(x, BaseModel):
            skip_list += [k for k, v in x.model_fields.items() if v.json_schema_extra and v.json_schema_extra.get('generate') == False]
            return {k: v for k, v in x.model_dump().items() if k not in skip_list}
        elif isinstance(x, Entity):
            skip_list += [k for k, v in x.model_fields.items() if v.json_schema_extra and v.json_schema_extra.get('generate') == False]
            return {k: v for k, v in x.object.dict().items() if k not in skip_list}
        elif isinstance(x, Collection):
            return [
                {k: v for k, v in y.dict().items() if k not in skip_list}
                for y in x.objects
            ]
        elif isinstance(x, str):
            return x
        elif isinstance(x, dict):
            return {k: self.parse(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [self.parse(y) for y in x]
        else:
            return x
    
    def render(self, t, x, **kwargs):
        input_template = JinjaTemplate(t.input_template)
        input = input_template.render(**x) if len(x) > 0 else ''
        output_template = JinjaTemplate(t.output_template)
        output = output_template.render(**x) if len(x) > 0 else ''
        vars = {
            **x,
            'instructions': t.instructions,
            'examples': self.render_examples(t),
            'format': self.render_format(t, x),
            'input': input,
            'output': output,
        }
        template = JinjaTemplate(t.template)
        output = template.render(**vars)
        return output
    
    def format_field(self, name, field, definitions, required):
        if name in ['id', 'type']:
            return None
        if field.get('generate', True) == False:
            return None
        description = field.get('description', '')
        options = ''
        metadata_keys = [
            'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum',
            'minLength', 'maxLength',
            'minItems', 'maxItems',
        ]
        metadata = {
            k: v for k, v in field.items()
            if k in metadata_keys
        }

        definition = None
        list_field = False
        if field.get('type') == 'array':
            list_field = True
            item_type = field.get('items', {}).get('type', None)
            if item_type is None:
                ref = field.get('items', {}).get('$ref', None)
                ref = ref.split('/')[-1]
                definition = definitions.get(ref, {})
                type_ = f'{definition.get("type")}[]'
            else:
                type_ = f'{item_type}[]'
            field = field.get('items', {})
        elif len(field.get('allOf', [])) > 0:
            ref = field.get('allOf')[0].get('$ref')
            ref = ref.split('/')[-1]
            definition = definitions.get(ref, {})
            type_ = f'{definition.get("type")}'
        elif field.get('$ref'):
            ref = field.get('$ref')
            ref = ref.split('/')[-1]
            definition = definitions.get(ref, {})
            type_ = f'{definition.get("type")}'
        else:
            type_ = field.get('type', 'str')
        
        if definition is not None and 'enum' in definition:
            if list_field:
                options += f'''
                Select any relevant options from: {", ".join(definition["enum"])}
                '''
            else:
                options += f'''
                Select one option from: {", ".join(definition["enum"])}
                '''

        if len(options) > 0:
            description += ' ' + options

        return {
            'name': name,
            'title': field.get('title', None),
            'type_': type_,
            'default': field.get('default', None),
            'description': description.strip(),
            'required': name in required,
            'metadata': metadata,
        }
    
    def render_format(self, t, x, **kwargs):
        if t.output is None or t.output == str:
            return ''
        
        output = json.loads(t.output)
        format_template = JinjaTemplate(t.format_template)
        if output.get('type', None) == 'array' and output.get('items', {}).get('type', None) == 'string':
            return format_template.render({
                'string_list_output': True,
            })

        list_output = False
        fields = []
        properties = {}
        if output.get('type', None) == 'array':
            properties = output.get('items', {}).get('properties', {})
            definitions = output.get('items', {}).get('$defs', {})
            required = output.get('items', {}).get('required', [])
            list_output = True
        elif output.get('type', None) == 'object':
            properties = output.get('properties', {})
            definitions = output.get('$defs', {})
            required = output.get('required', [])
        
        for name, property in properties.items():
            f = self.format_field(name, property, definitions, required)
            fields += [f]
        
        return format_template.render({
            'fields': [field for field in fields if field is not None], 
            'list_output': list_output,
        })
    
    def render_examples(self, t, **kwargs):
        if t.examples is None or len(t.examples) == 0:
            return ''
        
        examples = [
            {
                'input': e.input,
                'output': e.output,
            }
            for e in random.sample(t.examples, min(len(t.examples), 3))
        ]
        example_template = JinjaTemplate(t.example_template)
        return '\n'.join([
            example_template.render(**e) for e in examples
        ])
    
    def process(self, session, t, x, output, allow_none=False, **kwargs):
        if t.output is None:
            return output
        if allow_none and output is None:
            return None
        out = json.loads(output)
        schema = model_to_json_schema(json.loads(t.output))
        if schema.get('type', None) == 'string' or (schema.get('type', None) == 'array' and schema.get('items', {}).get('type', None) == 'string'):
            return out
        entities = create_entity_from_schema(schema, out, session=session, base=Entity)
        return entities
    
    def dict(self):
        return {
            'id': self.id,
            'type': 'template',
            'name': self.name or None,
            'instructions': self.instructions,
            'input': self.input,
            'output': self.output,
        }
    
    def __call__(self, session, t, x, llm, **kwargs):
        return self.forward(session, t, x, llm, **kwargs)
    
    def forward(self, session, t, x, llm, context=None, history=None, retries=3, dryrun=False, allow_none=False, **kwargs):
        if retries and retries <= 0:
            e = MaxRetriesExceeded(f'{t.name} failed to forward {x}')
            logger.error(e)
            raise e
        
        if dryrun:
            logger.debug(f'Dryrun: {t.output}')
            llm = MockLLM(output=t.output)
        
        px = self.parse(x)
            
        prompt_input = self.render(t, {'input': px})

        try:
            response = llm.generate(prompt_input, context=context or t.context, history=history)
        except openai.APIError as e:
            logger.error(f'LLM generation failed: {e}')
            time.sleep(2)
            if retries <= 1:
                raise e
            return self.forward(session, t, x, llm, retries=retries, **kwargs)
        except Exception as e:
            logger.error(f'Failed to generate {x}: {e}')
            if retries <= 1:
                raise e
            return self.forward(session, t, x, llm, retries=retries-1, **kwargs)

        try:
            response.content = self.process(session, t, px, response.raw, allow_none=allow_none, **kwargs)
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f'Output validation failed: {e}')
            if retries <= 1:
                raise e
            return self.forward(session, t, x, llm, retries=retries-1, **kwargs)
        except json.JSONDecodeError as e:
            logger.warning(f'Failed to decode JSON from {e}')
            if retries <= 1:
                raise e
            return self.forward(session, t, x, llm, retries=retries-1, **kwargs)
        except Exception as e:
            logger.error(f'Failed to forward {x}: {e}')
            if retries <= 1:
                raise e
            return self.forward(session, t, x, llm, retries=retries-1, **kwargs)
        return response
