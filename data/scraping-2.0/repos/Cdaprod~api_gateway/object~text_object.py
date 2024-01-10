from langchain.schema.runnable import Object, Text

markdown_schema = Object(
    id="markdown",
    description="Content of a Markdown file",
    attributes=[
        Text(
            id="content",
            description="The text content of a Markdown file.",
        )
    ],
    examples=[
        ("# Title\nThis is a markdown file.", [{"content": "# Title\nThis is a markdown file."}])
    ]
)

python_schema = Object(
    id="python",
    description="Content of a Python file",
    attributes=[
        Text(
            id="code",
            description="The code content of a Python file.",
        )
    ],
    examples=[
        ("def hello():\n    print('Hello, world!')", [{"code": "def hello():\n    print('Hello, world!')"}])
    ]
)

json_schema = Object(
    id="json",
    description="Content of a JSON file",
    attributes=[
        Text(
            id="json_string",
            description="The JSON string content of a JSON file.",
        )
    ],
    examples=[
        ('{"key": "value"}', [{"json_string": '{"key": "value"}'}])
    ]
)

yaml_schema = Object(
    id="yaml",
    description="Content of a YAML file",
    attributes=[
        Text(
            id="yaml_string",
            description="The YAML string content of a YAML file.",
        )
    ],
    examples=[
        ("key: value", [{"yaml_string": "key: value"}])
    ]
)