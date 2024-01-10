import enum
import importlib
import inspect
import json
import pkgutil

from langchain import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import logging
import json


def get_class_attributes(cls):
    # Filter to get only attributes with type annotations
    attributes = {
        name: attr
        for name, attr in inspect.getmembers(cls)
        if not name.startswith("_") and not inspect.isroutine(attr)
    }

    # Optionally, filter out attributes without type annotations
    # attributes = {name: attr for name, attr in attributes.items() if hasattr(attr, '__annotations__')}

    return attributes


def get_langchain_example_from_llm():
    prompt_template = f"""You\'re a helpful assistant. I\'ll give you example code that uses some Python libraries I\'m interested in, and I want you to provide me with a code example of RAG on langchain with the OpenAI LLM and AstraDB as my vector store. Don\'t give me any explanation or description. Just give me the code.

EXAMPLES:


EXAMPLE NOTEBOOK 1:

RAG
Let’s look at adding in a retrieval step to a prompt and LLM, which adds up to a “retrieval-augmented generation” chain

!pip install langchain openai faiss-cpu tiktoken

from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

vectorstore = FAISS.from_texts(
    [\"harrison worked at kensho\"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

template = \"\"\"Answer the question based only on the following context:
{{context}}

Question: {{question}}
\"\"\"
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = (
    {{\"context\": retriever, \"question\": RunnablePassthrough()}}
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke(\"where did harrison work?\")

\'Harrison worked at Kensho.\'

template = \"\"\"Answer the question based only on the following context:
{{context}}

Question: {{question}}

Answer in the following language: {{language}}
\"\"\"
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {{
        \"context\": itemgetter(\"question\") | retriever,
        \"question\": itemgetter(\"question\"),
        \"language\": itemgetter(\"language\"),
    }}
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke({{\"question\": \"where did harrison work\", \"language\": \"italian\"}})

\'Harrison ha lavorato a Kensho.\'


EXAMPLE NOTEBOOK 2:

# Get Wikipedia data

 !wget https://raw.githubusercontent.com/GeorgeCrossIV/Langchain-Retrieval-Augmentation-with-CASSIO/main/20220301.simple.csv

Import the 20220301.simple wikipedia from the CSV file

import pandas as pd

data = pd.read_csv(\'20220301.simple.csv\')

There are 10,000 entries in the Wikipedia data file. We\'ll reduce the dataset to 10 rows for this demo. It takes a while to process the data; however, feel free to increase the number of rows for future demo runs.

data = data.head(10)
data = data.rename(columns={{\'text \': \'text\'}})
data

We will execute queries against the [Andouille](https://simple.wikipedia.org/wiki/Andouille) Wikipedia entries later in this demo. The Wikipedia data used in this demo is from a snapshot in time, stored in a CSV file. Below is the text of the Wikipedia record that will be processed.

data.iloc[9][\'text\']

## Compute vector embedding using MiniLM

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import AstraDB as AstraDB
embeddings = HuggingFaceEmbeddings(model_name=\"all-MiniLM-L12-v2\")

### Setup AstraDB vector store instance for LangChain

cassandraVectorStore = AstraDB(
    embedding=embeddings,
    collection_name=ASTRA_COLLECTION,
    token=ASTRA_DB_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
)

# Build table with embeddings

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


mySplitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=120)

def create_vector_index(row, myCassandraVStore):
  metadata = {{
    \'url\': row[\'url\'],
    \'title\': row[\'title\']
  }}
  page_content = row[\'text\']

  wikiDocument = Document(
      page_content=page_content,
      metadata=metadata
  )
  wikiDocs = mySplitter.transform_documents([wikiDocument])
  myCassandraVStore.add_documents(wikiDocs)

for index, row in data.iterrows():
  create_vector_index(row, cassandraVectorStore)

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

model_id = \"google/flan-t5-xxl\"

credentials = {{
    \"url\"    : f\"https://{{IBM_REGION}}.ml.cloud.ibm.com\",
    \"apikey\" : apiSecret
}}

gen_parms = {{
    \"DECODING_METHOD\" : \"greedy\",
    \"MIN_NEW_TOKENS\" : 1,
    \"MAX_NEW_TOKENS\" : 50
}}

model = Model( model_id, credentials, gen_parms, PROJECT_ID )
lang_chain_compatible_model = WatsonxLLM(model=model)


results = cassandraVectorStore.similarity_search(\"What is Andouille?\", 1)

def print_results(exampleRows):
  for row in exampleRows:
    print(f\"\"\"{{row.page_content}}\n\"\"\")

print_results(results)

results = cassandraVectorStore.similarity_search(\"What temperature should Andouille be cooked?\", 3)
print_results(results)

We now have the relevant context that we can leverage to answer our question, but it\'s not super readable. Let\'s use IBM\'s watsonx chat completion API to provide a more readable answer.

First, let\'s prep the prompt.

The prompt will be prepared like this behind the scenes by LangChain:

```
def prepare_prompt(context, question):

  context_as_string =\'\n\'.join([row.body_blob for row in context])
  prompt_template = f\"\"\"Use the following pieces of context to answer the question at the end. If you don\'t know the answer, just say that you don\'t know, don\'t try to make up an answer.

  {{context_as_string}}

  Question: {{question}}
  Helpful Answer:\"\"\"
  return prompt_template

prompt = prepare_prompt(results, question)
```
(This prompt template was derived from [this LangChain module](https://github.com/devinbost/langchain/blob/f35db9f43e2301d63c643e01251739e7dcfb6b3b/libs/langchain/langchain/chains/question_answering/stuff_prompt.py#L10).)

from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(
        llm=lang_chain_compatible_model,
        chain_type=\"stuff\",
        retriever=cassandraVectorStore.as_retriever(),
        return_source_documents=True
)

### Create helper function to format the output:

def format_output(lc_result):
  print(f\"QUERY: {{lc_result[\'query\']}}\")
  print(f\"RESPONSE: {{lc_result[\'result\']}}\")
  print(f\"\n\nSOURCE DOCUMENTS:\n\")
  for doc in lc_result[\"source_documents\"]:
    print(f\"TITLE: {{doc.metadata[\'title\']}}\")
    print(f\"URL: {{doc.metadata[\'url\']}}\")

### Run watsonx via LangChain on a given question

Now that we have a prepared prompt and the relevant context, let\'s ask watsonx to give us a more readable answer.

output = chain({{\"query\":\"What temperature should Andouile be cooked?\"}})
format_output(output)

Notice that if we were asking a more difficult question across more source documents, the RAG integration would pull insights from muliple sources to give a more complete summary answer.

Regardless, let\'s compare our results to the results of running the model without the context provided by vector search and RAG.

import json

generated_response = model.generate(\"What temperature should Andouile be cooked?\", None )
print( json.dumps( generated_response[\"results\"][0][\"generated_text\"], indent=2 ) )

The \"\u00b0\" is unicode for the ° character, so the model without RAG gave us 190°C, which is 374 degrees Fahrenheit - quite different from the temperature provided by retrieval augmented generation!

374 degrees Fahrenheit might make sense for a more typical meat cooking temperature, but the critical differentiator here is that Andouille sausage is a smoked sausage by definition, which is clear from the context we retrieve via vector search. So, the correct answer was actually obtained when we leveraged vector search to provide the context necessary for watsonx to give us the desired answer.

### RAG is a powerful tool

To get more information, please reach out to us at Datastax or contact your IBM rep for more information about how we can partner with you to leverage RAG for your Generative AI use case.
[Check out more at our partnership page.](https://www.datastax.com/partners/ibm)"""
    chain = build_prompt_from_template(prompt_template)
    result = chain.invoke({})


def get_class_attributes_from_llm(cls):
    prompt_template = """You're a helpful assistant. When given the following Python source code, give me the attributes 
mentioned in the provided code and comments (if any) as a JSON object with the name, type, and default (if any) of each 
attribute. For example, from the following CODE (including comments), return the JSON provided. Don't provide any explanation. Format the results as JSON only so that I can parse it with the json.loads method in Python.
Also, prefer using defaults in the comments over the code. For example, if a comment specifies model_name = "sentence-transformers/all-mpnet-base-v2" but the code later specifies model_name: str = DEFAULT_MODEL_NAME, 
then the type is str, but the default is "sentence-transformers/all-mpnet-base-v2".
Additionally, if a comment specifies model_kwargs = {{'device': 'cpu'}} but the code shows model_kwargs: Dict[str, Any] = Field(default_factory=dict)
then the type is Dict[str, Any] but the default is {{'device': 'cpu'}}.
If no default is available for an attribute, set it to "None".
Also, give the outputs without backticks or markdown. I just want raw text.

EXAMPLE CODE:

class HuggingFaceEmbeddings(BaseModel, Embeddings):
\"\"\"HuggingFace sentence_transformers embedding models.

To use, you should have the ``sentence_transformers`` python package installed.

Example:
    .. code-block:: python

        from langchain.embeddings import HuggingFaceEmbeddings

        model_name = \"sentence-transformers/all-mpnet-base-v2\"
        model_kwargs = {{'device': 'cpu'}}
        encode_kwargs = {{'normalize_embeddings': False}}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
\"\"\"

client: Any  #: :meta private:
model_name: str = DEFAULT_MODEL_NAME
\"\"\"Model name to use.\"\"\"
cache_folder: Optional[str] = None
\"\"\"Path to store models. 
Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable.\"\"\"
model_kwargs: Dict[str, Any] = Field(default_factory=dict)
\"\"\"Key word arguments to pass to the model.\"\"\"
encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
\"\"\"Key word arguments to pass when calling the `encode` method of the model.\"\"\"
multi_process: bool = False
\"\"\"Run encode() on multiple GPUs.\"\"\"

def __init__(self, **kwargs: Any):
    \"\"\"Initialize the sentence_transformer.\"\"\"
    super().__init__(**kwargs)
    try:
        import sentence_transformers

    except ImportError as exc:
        raise ImportError(
            \"Could not import sentence_transformers python package. "
            "Please install it with `pip install sentence_transformers`.\"
        ) from exc

    self.client = sentence_transformers.SentenceTransformer(
        self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
    )

class Config:
    \"\"\"Configuration for this pydantic object.\"\"\"

    extra = Extra.forbid

def embed_documents(self, texts: List[str]) -> List[List[float]]:
    \"\"\"Compute doc embeddings using a HuggingFace transformer model.

    Args:
        texts: The list of texts to embed.

    Returns:
        List of embeddings, one for each text.
    \"\"\"
    import sentence_transformers

    texts = list(map(lambda x: x.replace("\n", " "), texts))
    if self.multi_process:
        pool = self.client.start_multi_process_pool()
        embeddings = self.client.encode_multi_process(texts, pool)
        sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
    else:
        embeddings = self.client.encode(texts, **self.encode_kwargs)

    return embeddings.tolist()

def embed_query(self, text: str) -> List[float]:
    \"\"\"Compute query embeddings using a HuggingFace transformer model.

    Args:
        text: The text to embed.

    Returns:
        Embeddings for the text.
    \"\"\"
    return self.embed_documents([text])[0]

EXAMPLE JSON:


[
    {{
        \"name\": \"client\",
        \"type\": \"Any\",
        \"default\": \"None\"
    }},
    {{
        \"name\": \"model_name\",
        \"type\": \"str\",
        \"default\": \"sentence-transformers/all-mpnet-base-v2\"
    }},
    {{
        \"name\": \"cache_folder\",
        \"type\": \"Optional[str]\",
        \"default\": \"None\"
    }},
    {{
        \"name\": \"model_kwargs\",
        \"type\": \"Dict[str, Any]\",
        \"default\": \"{{'device': 'cpu'}}\"
    }},
    {{
        \"name\": \"encode_kwargs\",
        \"type\": \"Dict[str, Any]\",
        \"default\": \"{{'normalize_embeddings': False}}\"
    }},
    {{
        \"name\": \"multi_process\",
        \"type\": \"bool\",
        \"default\": \"False\"
    }}
]

ACTUAL CODE:
{source_code}

ACTUAL JSON:
            """
    chain = build_prompt_from_template(prompt_template)
    source_code = inspect.getsource(cls)
    result = chain.invoke({"source_code": source_code})

    cleaned_result = remove_json_formatting(result)
    logging.info(f"Got result from LLM on class attributes: {cleaned_result}")

    json_result = json.loads(cleaned_result)

    return json_result


def build_prompt_from_template(prompt_template):
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = OpenAI(model_name="gpt-4-1106-preview")  # "gpt-3.5-turbo-1106")
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    return chain


def remove_json_formatting(input_string: str):
    # Check if the string starts with ```json and ends with ```
    if input_string.startswith("```json") and input_string.endswith("```"):
        # Remove the ```json at the start and ``` at the end
        return input_string[len("```json") : -len("```")].strip()
    else:
        # Return the original string if it doesn't start with ```json and end with ```
        return input_string


def get_class_method_params(cls):
    # Dictionary to store method names and their parameters
    methods_with_params = {}

    # Iterate over all members of the class
    for name, member in inspect.getmembers(cls):
        if not name.startswith("_"):
            # Check if the member is a method
            if inspect.isfunction(member) or inspect.ismethod(member):
                # Get the signature of the method
                signature = inspect.signature(member)
                # Store the method name and its parameters
                methods_with_params[name] = [
                    param.name
                    for param in signature.parameters.values()
                    if param.name != "self"
                ]
    return methods_with_params


def get_importable_classes(module_name):
    def walk_modules(module):
        classes = {}
        for loader, modname, ispkg in pkgutil.walk_packages(
            module.__path__, prefix=module.__name__ + "."
        ):
            try:
                sub_module = importlib.import_module(modname)
                if ispkg:
                    classes.extend(walk_modules(sub_module))
                else:
                    for name, obj in inspect.getmembers(sub_module, inspect.isclass):
                        if obj.__module__ == sub_module.__name__ and not issubclass(
                            obj, enum.Enum
                        ):
                            classes[f"{obj.__module__}.{obj.__name__}"] = obj
            except Exception as e:
                print(f"Error importing module {modname}: {e}")
                continue
        return classes

    try:
        main_module = importlib.import_module(module_name)
        return walk_modules(main_module)
    except Exception as e:
        print(f"Error importing main module {module_name}: {e}")
        return []
