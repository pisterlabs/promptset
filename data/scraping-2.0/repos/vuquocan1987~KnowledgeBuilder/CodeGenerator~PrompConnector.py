import openai
from dataclasses import dataclass
import yaml


ROOT_PATH = ""
def connectToChatGPT():
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "give me a short joke"},
        {"role": "user", "content": "tell me 1 sentence joke?"}
    ]
)

class PromptCreator:
    def __init__(self) -> None:
        self.guided_prompt = ""
        # self.get_input = self.get_input(self.guided_prompt)
    def get_input(self):
        raise NotImplementedError()
    def get_output(self):
        raise NotImplementedError()

@dataclass
class Service_Specification:
    name: str
    description: str
    openapi_spec: str
    deployment_spec: str
    prometheus_spec: str
    service_spec: str


SERVICE_LIST_FILE_PATH = "ServicesList.yaml"
def get_service_specification():
    with open(SERVICE_LIST_FILE_PATH) as f:
        service_list = yaml.load(f, Loader=yaml.FullLoader)
        service_specification_list = []
        for service in service_list["MicroServices"]:
            servce_name = service["Name"]
            service_description = service["Description"]
            service_open_api_path = f"{servce_name}-openapi.yaml"
            with open(service_open_api_path) as f:
                service_open_api_spec = f.read()
            service_deployment_path = f"{servce_name}-deployment.yaml"
            service_service_path = f"{servce_name}-service.yaml"
            service_prometheus_path = f"{servce_name}-prometheus.yaml"
            with open(service_deployment_path) as f:
                service_deployment_spec = f.read()
            with open(service_service_path) as f:
                service_service_spec = f.read()
            with open(service_prometheus_path) as f:
                service_prometheus_spec = f.read()
            service_specification = Service_Specification(servce_name, service_description, service_open_api_spec, service_deployment_spec, service_prometheus_spec,service_service_spec)  
            service_specification_list.append(service_specification)
    return service_specification_list



class SpecificationPromptCreator(PromptCreator):
    def __init__(self) -> None:
        super().__init__()

        self.guided_prompt = """
                On the next prompt I'll outline the specifications of our deployed microservices system and introduce a new requirement for the program. Your task is to provide the OpenAPI specification for a new microservice addition to the cluster.
Specifications detail the services active in the cluster. Each service comprises:

            Name: Identifying the service.
            Description: Elucidating the service's functionality.
            OpenAPI Specification: This is the YAML-formatted blueprint of the service. For instance, a login.yaml file suggests two services in the cluster: the login and browsing services.
            Deployment Specification: How the service is deployed.
            Prometheus Specification: Monitoring parameters for the service.
            Service Specification: How the service is accessed and related configurations.
The new requirement will be expressed in plain language. For example, "Introduce a service that adds two numbers." Based on the existing system and new requirements, your role is to adapt and provide the updated OpenAPI specification for the new service.
Give the output in the form of executable python code to create those files. just give me the code and nothing else.
        """

    def get_input(self):
        pass
            
    def get_output(self,new_requirement):
        service_specification_list = get_service_specification()
        promt_build = f"The system contains {len(service_specification_list)} services: \n"
        for service_specification in service_specification_list:
            promt_build += f"Service name: {service_specification.name}\n"
            promt_build += f"Service description: {service_specification.description}\n"
            promt_build += f"Service openapi spec file : {service_specification.openapi_spec}\n"
            promt_build += f"Service deployment spec file: {service_specification.deployment_spec}\n"
            promt_build += f"Service prometheus spec file: {service_specification.prometheus_spec}\n"
            promt_build += f"Service service spec file: {service_specification.service_spec}\n"

        promt_build += f"New requirement: f{new_requirement}\n"
        
        return promt_build
    
def get_new_specification_file():
    pass
            
