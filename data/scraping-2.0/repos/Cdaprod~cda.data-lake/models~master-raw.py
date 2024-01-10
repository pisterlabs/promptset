from pydantic import BaseModel, Field, root_validator, SecretStr, HttpUrl, ValidationError
from typing import Any, List, Optional, Dict, Union
from datetime import datetime
import requests


### ----------------- ExtractTransformLoad

# Define connection settings that could apply to any data source/storage
class ConnectionDetails(BaseModel):
    type: str  # E.g., 'S3', 'database', 'api', etc.
    identifier: str  # Name of the bucket, database, endpoint, etc.
    credentials: Dict[str, str]  # Could include tokens, keys, etc.
    additional_config: Optional[Dict[str, Any]]  # Any other necessary configuration

# Define a generic source model for extraction
class Source(BaseModel):
    source_id: str
    connection_details: ConnectionDetails
    data_format: Optional[str] = None  # E.g., 'csv', 'json', 'parquet', etc.
    extraction_method: Optional[str] = None  # E.g., 'full', 'incremental', etc.
    extraction_query: Optional[str] = None  # SQL query, API endpoint, etc.

# Define a generic transformation model
class Transformation(BaseModel):
    transformation_id: str
    description: Optional[str] = None
    logic: Optional[str] = None  # Reference to a transformation script or function
    dependencies: Optional[List[str]] = []  # IDs of transformations this one depends on

# Define a generic destination model for loading
class Destination(BaseModel):
    destination_id: str
    connection_details: ConnectionDetails
    data_format: Optional[str] = None

# Define a data lifecycle model
class DataLifecycle(BaseModel):
    stage: str  # E.g., 'raw', 'transformed', 'aggregated', etc.
    retention_policy: Optional[str] = None
    archival_details: Optional[str] = None
    access_permissions: Optional[List[str]] = None  # E.g., 'read', 'write', 'admin', etc.

# Define a job control model for ETL orchestration
class JobControl(BaseModel):
    job_id: str
    schedule: Optional[str] = None  # Cron expression for job scheduling
    dependencies: Optional[List[str]] = []  # IDs of jobs this one depends on
    alerting_rules: Optional[Dict[str, Any]] = None  # Alerting configuration

# Define a quality validation model
class QualityValidation(BaseModel):
    checks: Optional[Dict[str, Any]] = None  # E.g., {'null_check': 'No null values'}
    thresholds: Optional[Dict[str, float]] = None  # E.g., {'accuracy': 99.5}
    validation_rules: Optional[Dict[str, Any]] = None  # Custom validation rules

# Define an audit model for tracking ETL jobs
class Audit(BaseModel):
    timestamps: Dict[str, datetime] = Field(default_factory=lambda: {'created_at': datetime.now(), 'modified_at': datetime.now()})
    user_info: Optional[Dict[str, Any]] = None
    operation_type: Optional[str] = None  # E.g., 'ETL Process', 'Data Import', etc.

# Define a performance model for monitoring ETL jobs
class Performance(BaseModel):
    metrics: Optional[Dict[str, Any]] = None  # E.g., {'runtime_seconds': 120}
    logs: Optional[Dict[str, List[str]]] = None  # E.g., {'errors': ['error1', 'error2']}
    bottlenecks: Optional[List[str]] = None

# Define the main ETL process model
class ETLProcess(BaseModel):
    source: List[Source]
    transformations: List[Transformation]
    destination: List[Destination]
    lifecycle: DataLifecycle
    job_control: JobControl
    quality_validation: QualityValidation
    audit: Audit
    performance: Performance

@root_validator(pre=True)
def validate_structure(cls, values):
    """
    Custom validation to ensure the ETL process structure is coherent.
    This could include checks like ensuring all dependencies exist within
    the process definition, or that the data formats between source and
    destination are compatible.
    """
    # Example validation: Check if transformation dependencies are valid
    transformations = values.get('transformations', [])
    transformation_ids = {t.transformation_id for t in transformations}
    for transformation in transformations:
        if any(dep not in transformation_ids for dep in transformation.dependencies):
            raise ValueError('Invalid transformation dependency.')
    return values
    
### ----------------- GitHub

# Define the Owner model
class Owner(BaseModel):
    name: str
    id: int
    type: str  # User or Organization

# Define the Repository model
class Repository(BaseModel):
    id: int
    node_id: str
    name: str
    full_name: str
    owner: Owner
    private: bool
    html_url: HttpUrl
    description: Optional[str]
    fork: bool
    url: HttpUrl
    created_at: datetime
    updated_at: datetime
    pushed_at: datetime
    git_url: str
    ssh_url: str
    clone_url: HttpUrl
    size: int
    stargazers_count: int
    watchers_count: int
    language: Optional[str]
    has_issues: bool
    has_projects: bool
    has_downloads: bool
    has_wiki: bool
    has_pages: bool
    forks_count: int
    mirror_url: Optional[HttpUrl]
    archived: bool
    disabled: bool
    open_issues_count: int
    license: Optional[str]
    allow_forking: bool
    is_template: bool
    topics: List[str]
    visibility: str  # 'public' or 'private'
    
### ----------------- LangchainREPO (older)

# Define the base class for Runnables
class Runnable(BaseModel):
    id: str
    description: Optional[str]
    entry_point: str  # The command or function to run the Runnable

# Define a model for RunnableBranch
class RunnableBranch(BaseModel):
    id: str
    description: Optional[str]
    conditions: List[str]  # LCEL expressions to evaluate conditions
    runnables: List[Runnable]  # List of Runnables corresponding to conditions

# Define a model for LLM Chains
class LLMChain(BaseModel):
    chain_id: str
    description: Optional[str]
    components: List[str]  # IDs of components that make up this chain
    runnables: List[Union[Runnable, RunnableBranch]]  # Runnables and RunnableBranches in this chain

# Define a model for Agents
class Agent(BaseModel):
    agent_id: str
    description: Optional[str]
    capabilities: List[str]  # A list of actions or operations the agent can perform

# Define a model for custom pipelines
class CustomPipeline(BaseModel):
    pipeline_id: str
    description: Optional[str]
    steps: List[str]  # A list of step IDs that make up the pipeline

# Define a model for a Langchain application
class LangchainApp(BaseModel):
    app_id: str
    description: Optional[str]
    llm_chains: List[LLMChain]
    agents: List[Agent]
    custom_pipelines: List[CustomPipeline]
    runnables: List[Runnable]
    metadata: Dict[str, Union[str, int, List[str]]] = Field(default_factory=dict)

# Define a model for the entire langchain repo
class LangChainRepo(BaseModel):
    repo_url: HttpUrl
    apps: List[LangchainApp]  # List of LangchainApps within this repo
    
# Define a model for the entire langchain repo
class LangChainRepo(BaseModel):
    repo_url: HttpUrl
    apps: List[LangchainApp]  # List of LangchainApps within this repo
    
    class Config:
        schema_extra = {
            "example": {
                "repo_url": "https://github.com/Cdaprod/cda.langchain",
                "apps": [
                    {
                        "app_id": "app1",
                        "description": "First langchain application",
                        "llm_chains": [
                            {
                                "chain_id": "chain1",
                                "description": "LLM Chain 1",
                                "components": ["comp1", "comp2"],
                                "runnables": [
                                    {
                                        "id": "runnable1",
                                        "description": "Runnable 1",
                                        "entry_point": "python scripts/run1.py"
                                    }
                                ]
                            }
                        ],
                        "agents": [
                            {
                                "agent_id": "agent1",
                                "description": "Agent 1",
                                "capabilities": ["analyze_sentiment"]
                            }
                        ],
                        "custom_pipelines": [
                            {
                                "pipeline_id": "pipeline1",
                                "description": "Custom Pipeline 1",
                                "steps": ["step1", "step2"]
                            }
                        ],
                        "runnables": [
                            {
                                "id": "runnable2",
                                "description": "Runnable 2",
                                "entry_point": "python scripts/run2.py"
                            }
                        ],
                        "metadata": {"key1": "value1"}
                    },
                    {
                        "app_id": "app2",
                        "description": "Second langchain application",
                        # ... similar fields as app1
                    }
                ]
            }
        }
        

class ClientConnection(BaseModel):
    service_name: str = Field(..., description="Name of the service to connect to")
    service_type: str = Field(..., description="Type of service (e.g., 'database', 'api', 'cloud_storage')")
    hostname: Optional[str] = Field(None, description="Hostname of the service")
    port: Optional[int] = Field(None, description="Port to use for the connection")
    username: Optional[str] = Field(None, description="Username for authentication")
    password: Optional[SecretStr] = Field(None, description="Password for authentication")
    api_key: Optional[SecretStr] = Field(None, description="API key for services that require it")
    database_name: Optional[str] = Field(None, description="Name of the database to connect to, if applicable")
    additional_params: Optional[Dict[str, str]] = Field(None, description="Additional parameters required for the connection")
    tools: Optional[List[str]] = Field(None, description="List of tools associated with the service")
    required_vars: Optional[List[str]] = Field(None, description="List of required environment variables for the service")

    class Config:
        min_anystr_length = 1  # Ensuring that strings are not empty
        anystr_strip_whitespace = True  # Stripping whitespace from strings
        schema_extra = {
            "example": {
                "service_name": "Example API Service",
                "service_type": "api",
                "hostname": "api.example.com",
                "port": 443,
                "username": "apiuser",
                "password": "supersecretpassword",
                "api_key": "exampleapikey",
                "additional_params": {
                    "param1": "value1",
                    "param2": "value2"
                },
                "tools": ["curl", "httpie"],
                "required_vars": ["API_HOST", "API_KEY"]
            }
        }
        



class MetastoreAsset(BaseModel):
    asset_id: str = Field(..., description="Unique identifier for the asset")
    asset_type: str = Field(..., description="Type of the asset (e.g., 'table', 'view', 'model')")
    description: Optional[str] = Field(None, description="Description of the asset")
    location: HttpUrl = Field(..., description="URL to the asset location")
    schema: Dict[str, str] = Field(..., description="Schema definition of the asset")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default_factory=datetime.now, description="Last update timestamp")
    lineage: Optional[List[str]] = Field(default=[], description="List of asset_ids that are predecessors to this asset")

class Metastore(BaseModel):
    metastore_id: str = Field(..., description="Unique identifier for the metastore")
    assets: Dict[str, MetastoreAsset] = Field(default_factory=dict, description="Dictionary of assets by asset_id")
    repository: HttpUrl = Field(..., description="URL of the GitHub repository serving as the metastore")

    def add_asset(self, asset: MetastoreAsset) -> None:
        """
        Adds a new asset to the metastore.
        """
        self.assets[asset.asset_id] = asset
        self.updated_at = datetime.now()

    def get_asset(self, asset_id: str) -> MetastoreAsset:
        """
        Retrieves an asset from the metastore by its asset_id.
        """
        return self.assets[asset_id]

    def remove_asset(self, asset_id: str) -> None:
        """
        Removes an asset from the metastore by its asset_id.
        """
        del self.assets[asset_id]
        self.updated_at = datetime.now()
        
        
### ----------------- LangchainRepoEngine  
 
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel

# Import your LangchainDataLakeSystem
from your_langchain_module import LangChainRepo, FeatureStore, Runnable, Chain, Pipeline, Agent, LangchainDataLakeSystem

# Define SQLAlchemy Base and models
SQLAlchemyBase = declarative_base()

class ItemTable(SQLAlchemyBase):
    __tablename__ = 'items'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, index=True)

class ItemModel(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None

DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Instantiate the LangChainRepo and LangchainDataLakeSystem
repo = LangChainRepo()
data_lake_system = LangchainDataLakeSystem(repo=repo)

# FastAPI app instance from LangchainDataLakeSystem
app = data_lake_system.app

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    # Create all tables in the database
    SQLAlchemyBase.metadata.create_all(bind=engine)
    # Here you can also set up LangChain components if needed

# FastAPI route to create an item
@app.post("/items/", response_model=ItemModel)
def create_item(item: ItemModel, db: Session = Depends(get_db)):
    db_item = ItemTable(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

# FastAPI route to get an item by id
@app.get("/items/{item_id}", response_model=ItemModel)
def get_item(item_id: int, db: Session = Depends(get_db)):
    db_item = db.query(ItemTable).filter(ItemTable.id == item_id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item

# Additional FastAPI routes can be defined here

# Now you can add LangChain components to your LangchainDataLakeSystem
# For example, creating and registering a new Runnable with the system
# ...

# After defining LangChain components and routes, the FastAPI app can serve both purposes.

# Integration of LangChain components into the FastAPI application
from your_langchain_module import LangChainRepo, FeatureStore, Runnable, Chain, Pipeline, Agent

# Assuming your_langchain_module is the directory where your LangChain classes reside

repo = LangChainRepo()

# Define and create an example LangChain app with its components
# Use the previously defined models for Runnable, Chain, Pipeline, Agent

# Continue with the rest of your LangChain logic
# ...
from typing import Dict, Type, Any
from fastapi import FastAPI
from pydantic import BaseModel, create_model

# Define the base classes for all LangChain components
class Runnable(BaseModel):
    id: str
    name: str
    description: str
    metadata: Dict[str, Any] = {}
    script: str
    input: str
    output: str = None
    status: str = "not started"
    error: str = None

    def run(self):
        try:
            self.status = "running"
            # Execute the script here, using `self.input` as input
            # You might use the `exec` function to execute the script, but be aware of the security implications
            exec(self.script)
            # Store the output in `self.output`
            self.status = "completed"
        except Exception as e:
            self.status = "error"
            self.error = str(e)

class Chain(BaseModel):
    id: str
    runnables: Dict[str, Runnable]
    name: str
    description: str
    metadata: Dict[str, Any] = {}

    def add_runnable(self, runnable: Runnable):
        self.runnables[runnable.id] = runnable

class Pipeline(BaseModel):
    id: str
    chains: Dict[str, Chain]
    name: str
    description: str
    metadata: Dict[str, Any] = {}

    def add_chain(self, chain: Chain):
        self.chains[chain.id] = chain

class Agent(BaseModel):
    id: str
    pipelines: Dict[str, Pipeline]
    name: str
    description: str
    metadata: Dict[str, Any] = {}

    def add_pipeline(self, pipeline: Pipeline):
        self.pipelines[pipeline.id] = pipeline

# Define the FeatureStore for storing and retrieving features
class FeatureStore:
    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {
            'runnables': {},
            'chains': {},
            'pipelines': {},
            'agents': {}
        }

    def add_feature(self, feature_type: str, name: str, feature: Any):
        if feature_type not in self.store:
            raise ValueError(f"Invalid feature type: {feature_type}")
        self.store[feature_type][name] = feature

    def get_feature(self, feature_type: str, name: str) -> Any:
        if feature_type not in self.store:
            raise ValueError(f"Invalid feature type: {feature_type}")
        return self.store[feature_type].get(name)

    def delete_feature(self, feature_type: str, name: str):
        if feature_type in self.store and name in self.store[feature_type]:
            del self.store[feature_type][name]

# Define the LangChainRepo class to manage LangChain applications
class LangChainRepo:
    def __init__(self):
        self.feature_store = FeatureStore()

    def create_app(self, name: str, components: Dict[Type[BaseModel], Dict[str, BaseModel]]):
        for component_type, instances in components.items():
            for instance_name, instance in instances.items():
                component_type_name = component_type.__name__.lower() + 's'
                self.feature_store.add_feature(component_type_name, instance_name, instance)

    def delete_app(self, name: str):
        # Delete or de-register an app logic
        pass

# Define your LangchainDataLakeSystem that uses LangChainRepo
class LangchainDataLakeSystem:
    def __init__(self, repo: LangChainRepo):
        self.repo = repo
        self.app = FastAPI()

    def setup_routes(self):
        # Setup FastAPI routes logic
        pass

# Example usage of the LangChainRepo
repo = LangChainRepo()

# Define and create an example LangChain app with its components
# We create dynamic Pydantic models for example purposes
ExampleRunnable = create_model('ExampleRunnable', id=(str, ...), name=(str, ...), description=(str, ...), script=(str, ...), input=(str, ...))
ExampleChain = create_model('ExampleChain', id=(str, ...), runnables=(Dict[str, ExampleRunnable], ...))
ExamplePipeline = create_model('ExamplePipeline', id=(str, ...), chains=(Dict[str, ExampleChain], ...))
ExampleAgent = create_model('ExampleAgent', id=(str, ...), pipelines=(Dict[str, ExamplePipeline], ...))

example_runnable = ExampleRunnable(id='runnable1', name='Example Runnable', description='This is an example runnable', script='print("Hello, world!")', input='input string')
example_chain = ExampleChain(id='chain1', runnables={'runnable1': example_runnable})
example_pipeline = ExamplePipeline(id='pipeline1', chains={'chain1': example_chain})
example_agent = ExampleAgent(id='agent1', pipelines={'pipeline1': example_pipeline})

repo.create_app('example_app', {
    ExampleRunnable: {'example_runnable': example_runnable},
    ExampleChain: {'example_chain': example_chain},
    ExamplePipeline: {'example_pipeline': example_pipeline},
    ExampleAgent: {'example_agent': example_agent},
})

# Now you can retrieve a component from the FeatureStore like so:
retrieved_runnable = repo.feature_store.get_feature('runnables', 'example_runnable')


# Now, your FastAPI app can use the LangchainDataLakeSystem's features.
# You can add routes for LangChain operations, and use the repo to manage apps and components.

### ----------------- FeatureStore

import importlib
import pkgutil
import sys
from typing import Dict, Any

# Your existing FeatureStore class
class FeatureStore:
    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {
            'runnables': {},
            'chains': {},
            'pipelines': {},
            'agents': {}
        }

    def add_feature(self, feature_type: str, name: str, feature: Any):
        if feature_type not in self.store:
            raise ValueError(f"Invalid feature type: {feature_type}")
        self.store[feature_type][name] = feature

    def get_feature(self, feature_type: str, name: str) -> Any:
        if feature_type not in self.store:
            raise ValueError(f"Invalid feature type: {feature_type}")
        return self.store[feature_type].get(name)


# Function to import all modules from a package and store in FeatureStore
def import_and_store(package_name: str, feature_store: FeatureStore):
    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        full_module_name = f'{package_name}.{module_name}'
        module = importlib.import_module(full_module_name)
        # Assuming each module has a class named 'Feature'
        feature_class = getattr(module, 'Feature', None)
        if feature_class is not None:
            feature_instance = feature_class()
            feature_store.add_feature('runnables', module_name, feature_instance)


# Usage
feature_store = FeatureStore()
import_and_store('your_package_name', feature_store)

# Now your feature_store has the 'Feature' instances from all modules in 'your_package_name'

### ----------------- BuildDataLake

from data_lake_schema import (
    Cdaprod,
    ClientConnection,
    LangChainRepo,
    Repository,
    Server,
    Metastore,
    MetastoreAsset,
)

@dataclass
class RepoConfig:
    repo_url: str
    apps: Optional[List[str]] = None
    name: Optional[str] = None

@dataclass
class ClientConfig:
    service_name: str
    hostname: str
    credentials: Dict[str, str]

class BuildDataLake:
    def __init__(self, cdaprod_config: Dict):
        # Initialize the Cdaprod instance with its config
        self.cdaprod = Cdaprod(**cdaprod_config)

    def register_repository(self, repo_config: RepoConfig):
        if repo_config.apps is not None:  # LangChainRepo is distinguished by the presence of 'apps'
            repo_instance = LangChainRepo(repo_url=repo_config.repo_url, apps=repo_config.apps)
            self.cdaprod.services.append({'LangChainRepo': repo_instance})
        else:
            repo_instance = Repository(repo_url=repo_config.repo_url, name=repo_config.name)
            self.cdaprod.services.append({'Repository': repo_instance})

    def register_client(self, client_config: ClientConfig):
        client_instance = ClientConnection(
            service_name=client_config.service_name,
            hostname=client_config.hostname,
            credentials=client_config.credentials
        )
        self.cdaprod.services.append({'ClientConnection': client_instance})

    def register_metastore_asset(self, asset_config: Dict):
        asset_instance = MetastoreAsset(**asset_config)
        self.cdaprod.metastore.add_asset(asset_instance)

    def build(self, repo_list: List[RepoConfig], client_list: List[ClientConfig], asset_list: List[Dict]):
        for repo_config in repo_list:
            self.register_repository(repo_config)

        for client_config in client_list:
            self.register_client(client_config)

        for asset_config in asset_list:
            self.register_metastore_asset(asset_config)

        # Additional setup tasks can be added here
        # Return the fully built Cdaprod instance
        return self.cdaprod

# Configuration for server, metastore, and the main repository
cdaprod_config = {
    'server': Server(ip='192.168.1.1', creds={'username': 'user', 'password': 'pass'}, connection_type='SSH'),
    'repository': 'https://github.com/Cdaprod/main-repo',
    'services': [],
    'metastore': Metastore(metastore_id='metastore1', assets={}, repository='https://github.com/Cdaprod/metastore-repo'),
}

# Configuration for repositories, clients, and assets
repo_list = [
    RepoConfig(repo_url='https://github.com/Cdaprod/langchain-repo', apps=['app1', 'app2']),
    RepoConfig(repo_url='https://github.com/Cdaprod/other-repo', name='NonLangchainRepo'),
]

client_list = [
    ClientConfig(service_name='DatabaseService', hostname='db.example.com', credentials={'username': 'user', 'password': 'pass'}),
    ClientConfig(service_name='ApiService', hostname='api.example.com', credentials={'api_key': 'key'}),
]

asset_list = [
    {'asset_id': 'asset1', 'asset_type': 'table', 'location': 'http://example.com/asset1', 'schema': {'field1': 'type1', 'field2': 'type2'}},
    {'asset_id': 'asset2', 'asset_type': 'model', 'location': 'http://example.com/asset2', 'schema': {'fieldA': 'typeA', 'fieldB': 'typeB'}},
]

# Instantiate the builder and build the data lake
builder = BuildDataLake(cdaprod_config)
cdaprod_instance = builder.build(repo_list, client_list, asset_list)

# You can now work with the cdaprod_instance as needed


### ----------------- GitHubAPI

import requests
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from datetime import datetime
from Repository_model import Repository, Owner

# Insert your previously defined Owner and Repository classes here

class GitHubAPI:
    BASE_URL = 'https://api.github.com'

    def __init__(self, access_token):
        self.access_token = access_token
        self.headers = {
            'Authorization': f'token {self.access_token}',
            'Accept': 'application/vnd.github.v3+json',
        }
    
    def create_webhook(self, repository: Repository, webhook_url: str, events: List[str], secret: Optional[str] = None):
        """
        Create a webhook for the given repository.
        :param repository: Repository object
        :param webhook_url: The payload URL to deliver the payload to
        :param events: The list of events that the hook is triggered for
        :param secret: An optional secret for the hook's payload
        """
        webhook_data = {
            'config': {
                'url': webhook_url,
                'content_type': 'json',
                'secret': secret
            },
            'events': events,
            'active': True
        }

        # Construct the URL for webhook creation
        url = f"{self.BASE_URL}/repos/{repository.owner.name}/{repository.name}/hooks"

        response = requests.post(url, headers=self.headers, json=webhook_data)
        if response.status_code == 201:
            print(f"Webhook created for {repository.full_name}")
            return response.json()
        else:
            print(f"Failed to create webhook for {repository.full_name}: {response.text}")
            response.raise_for_status()


### -----------------



### -----------------




### -----------------


