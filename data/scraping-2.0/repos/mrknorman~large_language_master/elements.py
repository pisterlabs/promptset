import json
import re

from logging import getLogger
from dataclasses import dataclass
from typing import List, Tuple, Self
from pathlib import Path

from tqdm import tqdm
from openai import OpenAI

def ensure_directory_exists(
        directory: Path
    ):

    if not isinstance(directory, Path):
        raise ValueError("directory is not Path")  
    elif not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

def to_snake_case(input_string : str):
    # Replace non-alphanumeric characters with an underscore
    s = re.sub(r'\W+', '_', input_string)

    # Convert to lowercase
    return s.lower()

@dataclass
class APIConfig:
    system_prompt : str
    model : str
    temperature : float
    client : OpenAI

    def __post_init__(self):
        if not isinstance(self.system_prompt, str):
            raise ValueError("system_prompt is wrong type!")
        if not isinstance(self.model, str):
            raise ValueError("model is wrong type!")
        if not isinstance(self.temperature, float):
            raise ValueError("temperature is wrong type!")

class ElementPrompt:
    arguments : dict = None
    template : str = None
    prompt : str = None
    response : str = None 

    api_config : APIConfig = None

    num_prompt_retries : int = 0
    max_num_prompt_retries : int = 10

    def __init__(
            self,
            arguments : dict,
            template : str,
            api_config : APIConfig,
            max_num_prompt_retries : int = 10
        ) -> None:

        self.logger = getLogger(__name__)

        self.arguments = arguments
        self.template = template

        self.api_config = api_config

        self.num_prompt_retries = 0
        self.max_num_prompt_retries = max_num_prompt_retries

        self.generate_prompt()
        self.acquire_response(self.prompt)

    def generate_prompt(
        self
        ) -> None:

        if not isinstance(self.arguments, dict):
            raise ValueError("prompt_arguments is wrong type!")
        elif not isinstance(self.template, str):
            raise ValueError("prompt_template is wrong type!")
        else:
            self.prompt = self.template.format(
                **self.arguments
            )

    def acquire_response(
            self,
            prompt : str
        ) -> None:

        if not isinstance(self.api_config, APIConfig):
            raise ValueError("Argument 'api_config' should be APIConfig!")
        else:
            self.response = self.check_response(
                self.get_response(prompt), 
                prompt,
            )
    
    def get_response(
            self, 
            prompt : str,
            num_retries : int = 0,
            retry_sleep_duration_seconds : float = 1.0,
            maximum_num_retries : int = 10
        ) -> str:

        try:
            response = self.api_config.client.chat.completions.create(
                model=self.api_config.model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        'role':'system', 
                        'content': self.api_config.system_prompt
                        
                    },
                    {
                        'role':'user', 
                        'content': prompt
                    }
                ],
                temperature=self.api_config.temperature
            )

            return response
        except Exception as e:
            num_retries += 1

            if num_retries > maximum_num_retries:
                raise TimeoutError("Maximum num API requests reached.")
            
            self.logger.warning(
                (f"Call to API failed, because of {e} retrying for the "
                f"{num_retries} time in {retry_sleep_duration_seconds} s")
            )
            sleep(retry_sleep_duration_seconds)

            return self.get_response(prompt, num_retries)

    def check_response(
        self,
        response : str,
        prompt : str
    ):
        try:
            response = json.loads(
                response.choices[0].message.content
            )
        except Exception as e:
            self.logger.warning("Error occoured coverting API response to json. Retrying")
            self.num_prompt_retries += 1

            if self.num_prompt_retries > self.max_num_prompt_retries:
                raise TimeoutError((
                    "API failed return request in required format"
                    f"after {self.max_num_prompt_retries} retries!"
                ))

            response = self.get_response(
                (f"Error in previous response: '{response}' for prompt: '{prompt}'."
                f" Issue: '{e}'."
                " Please provide a response in valid JSON format as required by python's json.loads function."
                " Non-JSON responses will lead to an error."
                )
            )

        return response

@dataclass
class ElementType():
    name : str
    summary_attributes : List[str]
    prompt_arguments : List[str]
    prompt_template : str
    prompt_focus : str
    prompt_story_length : str
    prompt_examples : str
    child_element_dict : dict = None
    affordances : List = None

    def __post_init__(self):
        if self.affordances is None:
            self.affordances = []
        if not isinstance(self.name, str):
            raise ValueError("name is not string!")
        elif not isinstance(self.summary_attributes, list):
            raise ValueError("summary_attributes is not list!")
        elif not isinstance(self.prompt_arguments, list):
            raise ValueError("prompt_arguments is not list!")

@dataclass
class NetworkElementType(ElementType):
    node_key : str = None
    vertex_key : str = None
    verticies_prompt_arguments : List[str] = None
    verticies_prompt_template : str = None
    capsule : str = None

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.node_key, str):
            raise ValueError("node_key is not string!")
        elif not isinstance(self.vertex_key, str):
            raise ValueError("vertex_key is not string!")
        elif not isinstance(self.verticies_prompt_arguments, list):
            raise ValueError("verticies_prompt_arguments is not list!")
        elif not isinstance(self.verticies_prompt_template, str):
            raise ValueError("verticies_prompt_template is not string!")

@dataclass
class VertexElementType(ElementType):
    node_key : str = None
@dataclass
class NodeElementType(ElementType):
    vertex_key : str = None

@dataclass
class CapsuleType(ElementType):
    vertex_key : str = None
    node_key : str = None

class Element():

    description : str = None
    path : Path = None
    api_config : APIConfig = None
    prompt_arguments : dict = None
    prompt_template : str = None

    prompt : str = None
    outline : dict = None
    summary : dict = None

    def __init__(
            self,
            name : str, 
            type : ElementType,
            api_config : APIConfig,
            description : str = None, 
            path : Path = None, 
            parent : Self = None,
            siblings : List[Self] = None
        ):

        self.name = name
        self.snake_name = to_snake_case(name)

        self.description = description
        self.type = type

        self.api_config = api_config

        self.affordances = self.type.affordances

        self.parent = parent
        if self.parent is not None:
            self.parent_name =  parent.name
        else:
            self.parent_name = "None"
        self.siblings = siblings
        self.children = {}

        if self.parent is not None:
            self.lineage = parent.context 
        else:
            self.lineage = ""

        self.summary_attributes = self.type.summary_attributes
        self.prompt_arguments = {
            key : getattr(self, key) for key in self.type.prompt_arguments
        }
        self.prompt_template = self.type.prompt_template

        self.child_element_dict = self.type.child_element_dict
        
        self.envisioned = False

        self.generate_path(path)
        self.initlize_affordances()

    def initlize_affordances(self):
        affordance_dict = {}
        for affordance in self.affordances:
            affordance_object = affordance(
                is_possible=True,
                parent=self
            )
            affordance_dict[affordance_object.name] = affordance_object

        self.affordances = affordance_dict

    def generate_path(self, path : Path):
        if path is not None:
            self.path = path / f"{self.type.name}s"  / self.snake_name / f"{self.snake_name}.json"
        elif self.parent is not None:
            self.path = self.parent.path.parent / f"{self.type.name}s"  / self.snake_name / f"{self.snake_name}.json"

    def assemble(self):
        self.envision()
        self.save()
        if len(self.children):
            self.assemble_children()

    def assemble_children(self):
        self.initilize_children()
        self.envision_children()

    def save(
            self, 
            additional_attributes : dict = None, 
            force_overwrite : bool = False,
        ):

        save_attributes = {
            "name" : self.name,
            "description" : self.description,
            "parent_name" : self.parent_name,
            "lineage" : self.lineage,
            "summary" : self.summary,
            "context" : self.context
        }
        save_attributes.update({"outline" : self.outline})

        if additional_attributes is not None:
            save_attributes.update(additional_attributes)

        # Convert save_attributes to JSON
        save_json = json.dumps(save_attributes, indent=4)

        # Setup file name:
        ensure_directory_exists(self.path.parent)

        # Write to file
        with open(self.path, 'w') as file:
            file.write(save_json)

    def load(self):

        if not isinstance(self.path, Path):
            raise ValueError("self.path is not Path")
        elif not self.path.exists():
            raise FileExistsError("self.path does not exist!")

        with open(self.path, 'r') as file:
            data = json.load(file)

        for key, value in data.items():
            setattr(self, key, value)

    def envision(
            self, 
            force_reinit : bool = False
        ):

        if force_reinit is not True and self.path.exists():
            self.load()
        else:
            self.generate_primary_outline()

        self.encorporate_outline()
        self.envision_summary()
        self.context = self.lineage + self.summary

        self.envisioned = True

    def generate_primary_outline(self):
        self.prompt = ElementPrompt(
            arguments=self.prompt_arguments,
            template=self.prompt_template,
            api_config=self.api_config
        )
        self.outline = self.prompt.response

    def encorporate_outline(self):
        for key, value in self.outline.items():
            setattr(self, key, value)

    def envision_summary(self):
        extracted_attibute_values= {
            attr: getattr(self, attr) for attr in self.summary_attributes if hasattr(self, attr)}
        
        self.summary = ""
        for attr, value in extracted_attibute_values.items():
            self.summary += f' - {attr}: {value}\n'

    def initilize_children(self):
        
        for key, value in self.child_element_dict.items():

            children_outline = getattr(self, f"{key}_outline")
            child_type_dict = {}

            for child_name, child_plan in children_outline.items():
                new_child = value(
                    name=child_name,
                    api_config=self.api_config,
                    parent=self,
                    **child_plan
                )
                child_type_dict[child_name] = new_child
                self.children[child_name] = new_child

            setattr(
                self, 
                key, 
                child_type_dict
            )

    def envision_children(self):

        for child in self.children.values():
            child_type_name = child.type.name
            break

        for child in tqdm(
                self.children.values(),
                desc=f"The DM is envisioning {child_type_name}s for {self.name}..."
            ):
        
            child.assemble()

class Network(Element):

    def __init__(
            self,
            name : str,
            type : ElementType,
            api_config : APIConfig, 
            **kwargs
        ):

        self.node_key = type.node_key
        self.vertex_key = type.vertex_key
        self.capsule = type.capsule
        self.verticies_prompt_arguments = type.verticies_prompt_arguments
        self.verticies_prompt_template = type.verticies_prompt_template

        super().__init__(
            name=name, 
            type=type,
            api_config=api_config,
            **kwargs
        )

    def save(
            self, 
            additional_attributes : dict = None
        ):

        additional_attributes = {
            "verticies_outline" : self.verticies_outline,
        }
        super().save(additional_attributes=additional_attributes)

    def envision(self, force_reinit : bool = False):
        super().envision(
            force_reinit=force_reinit,
        )

        if force_reinit is not True and self.path.exists():
            self.load()
        else:
            self.generate_vertices()
            self.verticies_prompt_arguments = {
                key : getattr(self, key) for key in self.verticies_prompt_arguments
            }
            self.envision_vertices_outline()

        if self.capsule is not None:
            self.capsule = self.capsule(
                name=f"{self.name}_capsule",
                api_config=self.api_config,
                parent=self
            )

            self.capsule.assemble()
            self.children[self.capsule.type.name] = self.capsule
            self.entrances = self.capsule.entrances
            self.verticies_outline.update(self.entrances)

        self.encorporate_vertex_outline()

    def ensure_vertex_validity(self):  
        network_dict = getattr(self, f"{self.node_key}_outline")

        for key, value in network_dict.items():
            for connected_key in value["connected_to"]:
                if key not in network_dict[connected_key]["connected_to"]:
                    network_dict[connected_key]["connected_to"].append(key)

    def get_all_verticies(self):
        """Returns a set containing all pairs of connected rooms."""
        network_dict = getattr(self, f"{self.node_key}_outline")
        verticies = set()

        for key, node in network_dict.items():
            for connected_key in node["connected_to"]:
                # Creating a sorted tuple to ensure uniqueness (e.g., (A, B) == (B, A))
                pair = tuple(sorted([key, connected_key]))
                verticies.add(pair)
        
        setattr(self, f"{self.vertex_key}_map", verticies)
    
    def generate_vertices(self):
        self.ensure_vertex_validity()
        self.get_all_verticies()

    def envision_vertices_outline(self):
        self.verticies_prompt = ElementPrompt(
            arguments=self.verticies_prompt_arguments,
            template=self.verticies_prompt_template,
            api_config=self.api_config
        )
        self.verticies_outline = self.verticies_prompt.response

    def encorporate_vertex_outline(self):
        setattr(
            self, 
            f"{self.vertex_key}_outline", 
            self.verticies_outline
        )
    
class Node(Element):

    def __init__(
            self,
            name : str,
            type : ElementType,
            connected_to : Tuple[str],
            external_connection : bool,
            api_config : APIConfig, 
            parent : Element = None,
            **kwargs
        ):

        if parent is not None:
            verticies_outline = getattr(
                parent, 
                f"{parent.vertex_key}_outline"
            ) 

            self.connected_verticies = []
            for key, value in verticies_outline.items():
                for connection in value["asymmetries"]:
                    if connection in connected_to:
                        self.connected_verticies.append(verticies_outline[key])
        
        super().__init__(
            name=name, 
            type=type,
            api_config=api_config,
            parent=parent,
            **kwargs
        )

        self.vertex_key = self.type.vertex_key
        self.connected_to = connected_to 

class Capsule(Element):

    def __init__(
            self,
            name : str,
            type : ElementType,
            parent : Element,
            api_config : APIConfig, 
            **kwargs
        ):
        
        self.vertex_key = type.vertex_key
        self.node_key = type.node_key
        self.parent = parent
        self.get_all_verticies()
        
        super().__init__(
            name=name, 
            type=type,
            parent=parent,
            api_config=api_config,
            **kwargs
        )
    
    def get_all_verticies(self):
        """Returns a set containing all nodes connected to exterior."""
        network_dict = getattr(self.parent, f"{self.node_key}_outline")

        self.connected_to = []
        for key, node in network_dict.items():
            if node["external_connection"]:
                self.connected_to.append(key)

class Vertex(Element):

    def __init__(
            self,
            name : str,
            type : ElementType,
            connected_to : Tuple[str],
            api_config : APIConfig, 
            **kwargs
        ):
        
        super().__init__(
            name=name, 
            type=type,
            api_config=api_config,
            **kwargs
        )

        self.node_key = self.type.node_key
        self.connected_to = connected_to