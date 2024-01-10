import os
from softtek_llm.chatbot import Chatbot
from softtek_llm.models import OpenAI
from softtek_llm.cache import Cache
from softtek_llm.vectorStores import PineconeVectorStore
from softtek_llm.embeddings import OpenAIEmbeddings
from softtek_llm.schemas import Filter
from dotenv import load_dotenv
import json

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
  raise ValueError("OPENAI_API_KEY not found in .env file")

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
if OPENAI_API_BASE is None:
  raise ValueError("OPENAI_API_BASE not found in .env file")

OPENAI_EMBEDDINGS_MODEL_NAME = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME")
if OPENAI_EMBEDDINGS_MODEL_NAME is None:
  raise ValueError("OPENAI_EMBEDDINGS_MODEL_NAME not found in .env file")

OPENAI_CHAT_MODEL_NAME = os.getenv("OPENAI_CHAT_MODEL_NAME")
if OPENAI_CHAT_MODEL_NAME is None:
  raise ValueError("OPENAI_CHAT_MODEL_NAME not found in .env file")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if PINECONE_API_KEY is None:
  raise ValueError("PINECONE_API_KEY not found in .env file")

PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
if PINECONE_ENVIRONMENT is None:
  raise ValueError("PINECONE_ENVIRONMENT not found in .env file")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
if PINECONE_INDEX_NAME is None:
  raise ValueError("PINECONE_INDEX_NAME not found in .env file")

vector_store = PineconeVectorStore(
  api_key=PINECONE_API_KEY,
  environment=PINECONE_ENVIRONMENT,
  index_name=PINECONE_INDEX_NAME,
)
embeddings_model = OpenAIEmbeddings(
  api_key=OPENAI_API_KEY,
  model_name=OPENAI_EMBEDDINGS_MODEL_NAME,
  api_type="azure",
  api_base=OPENAI_API_BASE,
)
cache = Cache(
  vector_store=vector_store,
  embeddings_model=embeddings_model,
)
model = OpenAI(
  api_key=OPENAI_API_KEY,
  model_name=OPENAI_CHAT_MODEL_NAME,
  api_type="azure",
  api_base=OPENAI_API_BASE,
  verbose=True,
)

filters_code = [
  Filter(
    type="ALLOW",
    case="Allow all code samples with any language",
  ),
  Filter(
    type="ALLOW",
    case="Allow all code samples with any language",
  )
]

filters_resume = [
  Filter(
    type="ALLOW",
    case="Allow all personal details, work details and resume details",
  ),
  Filter(
    type="ALLOW",
    case="Allow all personal details, work details and resume details",
  )
]

chatbot_code = Chatbot(
  model=model,
  description="""You are an analizing tool which specializes in the extraction of used data structures, libraries and tools in code snippets.
    Your job is to analyze the given code and find the mentioned information, returning a JSON object with the findings.
    You should include a 1-10 score on an overall proeficiency in the language and a 0-5 in difficulty.
    The JSON format should be as following:
    'results': {
      'data_structures': [],
      'libraries': [],
      'tools': [],
      'score': ,
      'difficulty':
    }""",
  filters=filters_code,
  cache=cache,
  verbose=True,
)

chatbot_resume = Chatbot(
  model=model,
  description="""You are an analizing tool which specializes in the extraction of programming languages used, skills and tools in resumes.
    Remember to only inlude programming languages in the JSON, not spoken languages. Also, always base your findings on the context of the current resume, not previous one.
    Your job is to analyze the given resume and find the mentioned information, returning a JSON object with the findings.
    The JSON format should be as following:
    'results': {
      'languages': [],
      'skills': [],
      'tools': []
    }""",
  filters=filters_resume,
  cache=cache,
  verbose=True,
)


def generateCodeSnippet(languages, skills, tools):
  chatbot_codeGenerator = Chatbot(
    model=model,
    description="""You are an code generator tool which generates code challenges based on different parameters.
      You should generate coding question based on the given parameters. Make sure to create problems with test 
      cases that are easy to understand and that are not too long.
      In this case, consider that the user has experience in """ + languages + skills + tools +
      """The JSON format should be as following:
      'results': {
        'problem': [],
        'test_cases': [],
        'language': ,
      }""",
    filters=filters_code,
    cache=cache,
    verbose=True,
  )

  response = ""
  flag = False

  while not flag:
    try:
      response = chatbot_codeGenerator.chat("Please generate the JSON with my coding challenge.", print_cache_score=False)
      parsed_response = response.message.content.replace("\n", "").replace("\'", "")
      print(parsed_response)
      response = json.loads(parsed_response)["results"]
      flag = True

      # UPLOAD TO MONGODB
    except:
      print("TRYING")

  return response


def analyzeCV(promt):
  response = ""
  flag = False

  while not flag:
    try:
      response = chatbot_resume.chat(promt, print_cache_score=False)
      # {"message":{"role":"assistant","content":"{\n  \"results\": {\n    \"languages\": [],\n    \"skills\": [],\n    \"tools\": []\n  }\n}"},"created":1695548263,"latency":812,"from_cache":false,"model":"gpt-35-turbo-16k","usage":{"prompt_tokens":99,"completion_tokens":24,"total_tokens":123},"additional_kwargs":{}}
      parsed_response = response.message.content.replace("\n", "").replace(" ", "").replace("\'", "")
      print(parsed_response)
      # "{results:{languages:[],skills:[],tools:[]}}"
      response = json.loads(parsed_response)["results"]
      flag = True

      # UPLOAD TO MONGODB
    except:
      print("TRYING")

  return response