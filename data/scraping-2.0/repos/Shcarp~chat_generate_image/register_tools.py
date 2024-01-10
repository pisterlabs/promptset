import inspect
import json
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from abc import abstractmethod, ABC
from types import GenericAlias
from typing import get_origin, Annotated

load_dotenv()

_PARAM_TYPE_MAP = {
    "str": "string",
    "int": "number",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object"
}

__FUNCTIONS__ = {}

client = OpenAI()

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

class StoreMedium(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def add(self, data):
        pass

    @abstractmethod
    def search(self, query):
        pass

    @abstractmethod
    def all(self):
        pass

class ChromadbStoreMedium(StoreMedium):
    def __init__(self, name: str, embedding: callable = None, parse = None):
        if embedding == None:
            raise TypeError("embedding must be a callable")
        
        super().__init__(name)
        self.parse = parse
        self.embedding = embedding
        self.db = chromadb.Client()
        self.collection = self.db.get_or_create_collection(name)
        
    def add(self, data):
        self.collection.add(
            documents=data["documents"],
            metadatas=data["metadatas"], 
            embeddings=data["embeddings"],
            ids=data["ids"]
        )

    def search(self, query, n=10):
        embeddings = self.embedding(query) if self.embedding != None else []
        tool_list = []
        source_list = self.collection.query(
            query_embeddings=[embeddings],
            n_results=n
        )
        for source in source_list["metadatas"]:
            tool_list.append(self.parse(source) if self.parse != None else source)
        return tool_list
    
    def all(self):
        tool_list = []
        source_list = self.collection.get()
        for source in source_list["metadatas"]:
            tool_list.append(self.parse(source) if self.parse != None else source)
        return tool_list

def parse_openai_tool(tool: dict):
    return {
        type: 'function',
        function: {
            "name": tool.name,
            "description": tool.description,
            "parameters": json.loads(tool.parameters)
        }
    }

def translate_into_openai_tool(config: dict):
    tool_name = config["name"]
    tool_description = config["description"]
    tool_params = config["params"]

    openai_params = {
        "type": "object",
        "properties": {},
        "required": []
    }

    for param in tool_params:
        param_name = param["name"]
        param_description = param["description"]
        param_type = _PARAM_TYPE_MAP[param["type"]]
        param_required = param["required"]

        openai_params["properties"][param_name] = {
            "type": param_type,
            "description": param_description
        }

        if param_required:
            openai_params["required"].append(param_name)

    openai_tool = {
        "name": tool_name,
        "description": tool_description,
        "parameters": json.dumps(openai_params)
    }

    return openai_tool

def register_tools(
    translation: callable = None, 
    tools_store: StoreMedium = ChromadbStoreMedium("tools", get_embedding),
    embedding: callable = None
):
    def register(func: callable):
        tool_name = func.__name__
        tool_description = inspect.getdoc(func).strip()
        python_params = inspect.signature(func).parameters
        tool_params = []
        for name, param in python_params.items():
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                raise TypeError(f"Parameter `{name}` missing type annotation")
            if get_origin(annotation) != Annotated:
                raise TypeError(f"Annotation type for `{name}` must be typing.Annotated")

            typ, (description, required) = annotation.__origin__, annotation.__metadata__
            typ: str = str(typ) if isinstance(typ, GenericAlias) else typ.__name__
            if not isinstance(description, str):
                raise TypeError(f"Description for `{name}` must be a string")
            if not isinstance(required, bool):
                raise TypeError(f"Required for `{name}` must be a bool")

            tool_params.append({
                "name": name,
                "description": description,
                "type": typ,
                "required": required
            })
        
        tool_def = {
            "name": tool_name,
            "description": tool_description,
            "params": tool_params
        }

        document = tool_def["description"]

        if translation:
            tool_def = translation(tool_def)

        add_data = {
            "embeddings": embedding(document) if embedding != None else [],
            "documents": document,
            "metadatas": tool_def,
            "ids": document
        }
    
        tools_store.add(add_data)
        
        __FUNCTIONS__[tool_name] = func
        
        return func
    
    return register

def get_tools_list(
    query = None, 
    tools_store: StoreMedium = ChromadbStoreMedium("tools", get_embedding),
    n=10
):
    if query == None:
        return tools_store.all()
    
    return tools_store.search(query, n)

def dispatch(data: dict):
    tool_name = data["tool"]
    tool_params = data["params"]

    if tool_name not in __FUNCTIONS__:
        raise ValueError(f"Tool `{tool_name}` not found")

    return __FUNCTIONS__[tool_name](**tool_params)

@register_tools(
    embedding=get_embedding, 
    translation=translate_into_openai_tool
)
def test_func_1(
    prompt: Annotated[str, "This is a prompt", True],
):
    """This is a test function"""
    print(prompt)

@register_tools(
    embedding=get_embedding, 
    translation=translate_into_openai_tool
)
def test_func_2(
    prompt: Annotated[str, "This is a prompt", True],
):
    """This is a test2 function"""
    print(prompt)


@register_tools(
    embedding=get_embedding, 
    translation=translate_into_openai_tool
)
def test_func_3(
    prompt: Annotated[str, "This is a prompt", True],
):
    """This is a test3 function"""
    print(prompt)


# print(get_tools_list())
