import logging
import json
import requests
import click
import torch
import shutil
import os
from pymongo import MongoClient
from flask_cors import CORS
from flask import Flask, jsonify, request
from bson import ObjectId
import subprocess
from pymongo import MongoClient
import json
import csv
# model
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY

import warnings

# Ignore all warnings (not recommended unless you're sure about it)
warnings.filterwarnings("ignore")

# Ignore specific category of warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)
mongo_connection_string = 'mongodb+srv://mrunal21:mrunal21@cluster0.eugjmpy.mongodb.net'
client = MongoClient(mongo_connection_string)
mongoDB = client['violatedData']
collection_datasets = mongoDB['datasets']
collection_customer = mongoDB['customer']
collection_blocked = mongoDB['blockedUsers']
collection_input = mongoDB['input']
collection_input2 = mongoDB['input2']
rules= {
    "users": [
      {
        "type": "admin",
        "two_factor_authentication": True,
        "multi_factor_authentication": True,
        "security_monitoring": True,
        "data_privacy_policy": True,
        "secure_file_uploads": True,
        "secure_file_uploads_policies": {
          "properties": {
            "secure_file_name": [".txt", ".csv", ".xlsx", ".pdf",".img",".png",".jpeg",".mp4"],
            "malware_scan": True,
            "audit_logging": True,
            "encryption": {
              "in_transit": True,
              "at_rest": True
            }
          }
        },
        "ssl_encryption_required": True,
        "permissions": ["read", "write", "delete", "create"],
        "explicite_allowed_resources": [
          "sensitve_data.txt",
          "sales.txt",
          "reports.txt",
          "product_info.txt"
        ],
        "other_resources": True
      },
      {
        "type": "employee",
  
        "two_factor_authentication": True,
        "multi_factor_authentication": False,
        "security_monitoring": True,
        "data_privacy_policy": True,
        "secure_file_uploads": True,
        "secure_file_uploads_policies": {
          "properties": {
            "secure_file_name": [".txt", ".csv", ".xlsx", ".pdf",".img",".png",".jpeg",".mp4"],
            "malware_scan": True,
            "audit_logging": True,
            "encryption": {
              "in_transit": True,
              "at_rest": True
            }
          }
        },
        "ssl_encryption_required": True,
        "permissions": ["read", "write", "create"],
        "explicite_allowed_resources": [
          "sales.txt",
          "reports.txt",
          "product_info.txt"
        ],
        "other_resources": False
      },
      {
        "type": "customer",
  
        "two_factor_authentication": True,
        "multi_factor_authentication": True,
        "security_monitoring": True,
        "data_privacy_policy": True,
        "secure_file_uploads": False,
        "secure_file_uploads_policies": {
          "properties": {
            "secure_file_name": [".txt", ".csv", ".xlsx", ".pdf",".img",".png",".jpeg",".mp4"],
            "malware_scan": True,
            "audit_logging": True,
            "encryption": {
              "in_transit": True,
              "at_rest": True
            }
          }
        },
        "ssl_encryption_required": True,
        "permissions": ["read"],
        "explicite_allowed_resources": ["product_info.txt", "userId_info.txt"],
        "other_resources": False
      }
    ]
}

policy_score={
    "policies": [
      {
        "name": "admin",
        "properties": {
          "two_factor_authentication": 9,
          "multi_factor_authentication": 9,
          "security_monitoring": 10,
          "data_privacy_policy": 10,
          "secure_file_uploads": 8,
          "secure_file_uploads_policies": {
            "secure_file_name": 7,
            "malware_scan": 8,
            "audit_logging": 9,
            "sandboxing": 8,
            "encryption": {
              "in_transit": 10,
              "at_rest": 9
            }
          },
          "ssl_encryption_required": 8,
          "permissions": 9,
          "explicite_allowed_resources": 7,
          "other_resources": 6
        }
      },
      {
        "name": "employee",
        "properties": {
          "two_factor_authentication": 8,
          "multi_factor_authentication": 5,
          "security_monitoring": 7,
          "data_privacy_policy": 7,
          "secure_file_uploads": 6,
          "secure_file_uploads_policies": {
            "secure_file_name": 5,
            "malware_scan": 6,
            "audit_logging": 6,
            "encryption": {
              "in_transit": 8,
              "at_rest": 7
            }
          },
          "ssl_encryption_required": 7,
          "permissions": 6,
          "explicite_allowed_resources": 5,
          "other_resources": 3
        }
      },
      {
        "name": "customer",
        "properties": {
          "two_factor_authentication": 7,
          "security_monitoring": 6,
          "data_privacy_policy": 5,
          "secure_file_uploads": 3,
          "secure_file_uploads_policies": {
            "secure_file_name": 4,
            "malware_scan": 6,
            "audit_logging": 5,
            "encryption": {
              "in_transit": 7,
              "at_rest": 7
            }
          },
          "ssl_encryption_required": 5,
          "permissions": 4,
          "explicite_allowed_resources": 4,
          "other_resources": 3
        }
      }
    ]
  }

device_type="cuda"
show_sources="True"

UPLOAD_FOLDER = 'System_generated_Logs/scripts/uidata/uploads'
UPLOAD_FOLDER_RULES = 'System_generated_Logs/scripts/uploaded_rules'
UPLOAD_FOLDER_AUDIO = 'Human_generated_Logs/data/audio'
app.config['UPLOAD_FOLDER_AUDIO'] = UPLOAD_FOLDER_AUDIO

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_RULES'] = UPLOAD_FOLDER_RULES 

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(UPLOAD_FOLDER_RULES):
    os.makedirs(UPLOAD_FOLDER_RULES)

os.chmod(UPLOAD_FOLDER, 0o755)


file_path = './System_generated_Logs/jsons/attacks/security_attacks.json'

with open(file_path, 'r') as json_file:
    security_attacks = json.load(json_file)

# For System Logs
#step 1 - get random instance
@app.route('/random_instance', methods=['GET'])
def get_random_instance():
    try:
      pipeline = [
        {"$sample": {"size": 1}}
      ]
      result_list = list(collection_datasets.aggregate(pipeline))
      random_instance = result_list[0]
      random_instance['_id'] = str(random_instance['_id'])
      logging.info('random_instance')    
      return jsonify(random_instance)
    except Exception as e:
      return jsonify({"error": str(e)}), 500   

#step 2 - get severity score
@app.route('/get_score_calculation', methods=['POST'])
def get_score_calculation():
    data_gsc = request.json
    score,graph_list, violated_policies = score_calculation(data_gsc)
    temp_data = {
        "score":score,
        "graph_list" : graph_list,
        "violated_policies":violated_policies
    }
    return jsonify(temp_data)

# step 2.1 
def score_calculation(instance):
    instance = detect_user(instance)
    graph_list=[]
    violated_policies = instance['violated_policies']
    violated_tags = list(violated_policies.keys())
    print(violated_tags)
    score = 0
    user_type = instance['type']
    l=len(violated_policies)
    if(l == 0):
        return 100

    for policy in policy_score['policies']:
        if policy['name'] == user_type:
            for violation in violated_policies:
                if violation in policy['properties']:
                    score += policy['properties'][violation]
                    graph_list.append(policy['properties'][violation])
    
    return score/l,graph_list, violated_tags

#step 2.2
def detect_user(instance):
    new_instance={}
    for key, value in instance.items():
        new_instance[key] = value
    
    violations=check_policy_violation(instance)
    
    new_instance["violated_policies"]=violations
    
    return new_instance

# step 2.3
def check_policy_violation(instance):
    violations = {}

    k=instance["type"]
    desired_users = [user for user in rules["users"] if user["type"] == k]

    
    # Iterate over the rules for each user type
    for user_rule in desired_users:
            
            if user_rule["two_factor_authentication"] != instance["two_factor_authentication"]:
                violations["two_factor_authentication"]=instance["two_factor_authentication"]
            if user_rule["multi_factor_authentication"] != instance["multi_factor_authentication"]:
                violations["multi_factor_authentication"]=instance["multi_factor_authentication"]    
            if user_rule["security_monitoring"] != instance["security_monitoring"]:
                violations["security_monitoring"]=instance["security_monitoring"]
            if user_rule["data_privacy_policy"] != instance["data_privacy_policy"]:
                violations["data_privacy_policy"]=instance["data_privacy_policy"]
            
            
 
            for i in user_rule["secure_file_uploads_policies"]:
                for j in user_rule["secure_file_uploads_policies"][i]:
                    fname="secure_file_uploads_policies"
                    fname=fname+"__"+i
                    fname=fname+"__"+j

                    if(j=="secure_file_name"):
                      last_exe=instance[fname].split('.')
                      extension="."+last_exe[-1]
                      if extension not in [".txt", ".csv", ".xlsx", ".pdf",".img",".png",".jpeg",".mp4"]:
                          violations[fname]=instance[fname]
                    if(j=="malware_scan" and user_rule["secure_file_uploads_policies"][i][j]!=instance[fname]):
                        violations[fname]=instance[fname]
                    if(j=="audit_logging" and user_rule["secure_file_uploads_policies"][i][j]!=instance[fname]):
                        violations[fname]=instance[fname]
                    if(j=="encryption"):
                      for k in user_rule["secure_file_uploads_policies"][i][j]:
                        fname="secure_file_uploads_policies"
                        fname=fname+"__"+i
                        fname=fname+"__"+j
                        fname=fname+"__"+k
                        if(user_rule["secure_file_uploads_policies"][i][j][k]!=instance[fname]):
                            violations[fname]=instance[fname]
    
            if "ssl_encryption_required" in user_rule and user_rule["ssl_encryption_required"] != instance["ssl_encryption_required"]:
                violations["ssl_encryption_required"]=instance["ssl_encryption_required"]
            
            if "permissions" in user_rule and any(permission not in user_rule["permissions"] for permission in instance["permissions"]):
                violations["permissions"]=instance["permissions"]
            if "explicite_allowed_resources" in user_rule and instance["explicite_allowed_resources"] not in user_rule["explicite_allowed_resources"]:
                violations["explicite_allowed_resources"]=instance["explicite_allowed_resources"]
   
    return violations

# step 3 - create a prompt
@app.route('/create_prompt', methods=['POST'])
def create_prompt():
    instance = detect_user(request.json)
    # print(instance)
    context = context_gen(instance)
    rules = rule_gen(instance)
    question=" Answer me that, are there any security violations in context based on rules? "
    data_prompt='Context: '+context+'\n'+'Rules: '+rules+'\n'+'Question: '+question+'\n'
    return jsonify(data_prompt)

# step 3.1 
def context_gen(instance):
    print(instance)
    explanation_paragraph = (
    f"The information displayed in the 'client' field is denoted as '{instance['client']}' and the timestamp is indicated by 'datetime' as '{instance['datetime']}'. The method used was '{instance['method']}' with a link labeled 'request' pointing to '{instance['request']}'.The source that referred the request is captured in 'referer' as '{instance['referer']}' and the originating device is identified by 'user_agent' as '{instance['user_agent']}'."
    f"Categorized as '{instance['type']}', this instance's two-factor authentication is {'enabled' if instance['two_factor_authentication'] else 'disabled'}, and multi-factor authentication is {'enabled' if instance['multi_factor_authentication'] else 'disabled'}. The utilization of 'security_monitoring' is {'enabled' if instance['secure_file_uploads'] else 'disabled'}, along with a data privacy policy that is {'enabled' if instance['data_privacy_policy'] else 'disabled'}. The setting for 'secure_file_uploads' is {'enabled' if instance['secure_file_uploads'] else 'disabled'}. The specified 'secure_file_name' is '{instance['secure_file_uploads_policies__properties__secure_file_name']}' and the malware scan feature is {'enabled' if instance['secure_file_uploads_policies__properties__malware_scan'] else 'disabled'}. The option for 'audit_logging' is {'enabled' if instance['secure_file_uploads_policies__properties__audit_logging'] else 'disabled'}."
    f"The status of 'Encryption in transit' is {'enabled' if instance['secure_file_uploads_policies__properties__encryption__in_transit'] else 'disabled'}, and 'encryption at rest' is {'enabled' if instance['secure_file_uploads_policies__properties__encryption__at_rest'] else 'disabled'}. Additionally, 'SSL encryption' is {'enabled' if instance['ssl_encryption_required'] else 'disabled'}. The permissions are listed as '{instance['permissions']}' and the explicitly allowed resources are '{instance['explicite_allowed_resources']}'. The availability of 'other_resources' is {'enabled' if instance['other_resources'] else 'disabled'}. The HTTP 'status_code' '{instance['status']}' reflects the specific status of the HTTP request."
    f"Notably, the 'violated_polices' section highlights that '{instance['violated_policies']}' policies were breached.")
    
    return explanation_paragraph

#step 3.2 
def rule_gen(instance):
    keys=find_violated_polices(instance).keys()
    rules={}
    for key in keys:
        rules[key]=security_attacks[key]
    # print(rules)
    rules=json.dumps(rules)
    return rules

#  step 3.3
def find_violated_polices(instance):
    violated_policies = instance['violated_policies']
    return violated_policies
 
# step 4.0 
def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            max_ctx_size = 3500
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)

        else:
            # The code supports all huggingface models that ends with GPTQ and have some variation
            # of .no-act.order or .safetensors in their HF repo.
            logging.info("Using AutoGPTQForCausalLM for quantized models")

            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            logging.info("Tokenizer loaded")

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    elif (
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=4000,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm

@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)

# step 4 - LLM
@app.route('/fetch_llm_response' , methods=['POST'])
def fetch_llm_response():
    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()
    
    # model_id = "TheBloke/OpenOrca-Platypus2-13B-GPTQ"
    # model_basename = "gptq_model-4bit-128g.safetensors"
    
    model_id = "TheBloke/OpenOrca-Platypus2-13B-GPTQ"
    model_basename = "gptq_model-4bit-128g.safetensors"

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.{context} {history} Question: {question} Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")
    llm = load_model(device_type, model_id=model_id, model_basename=model_basename)
    logging.info(f"load model called")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    logging.info(f"QA instance made")
    res = qa(request.data.decode('utf-8'))
    answer, docs = res["result"], res["source_documents"]
    logging.info(f"QA analyzed answer")
    return jsonify({"answer":answer})

# for block user feature
@app.route('/block_user', methods=['POST'])
def block_user(): 
    temp_data = request.json
    collection_blocked.insert_one(temp_data)
    document_id = ObjectId(temp_data['id'])
    collection_datasets.delete_one({"_id": (document_id)})
    return jsonify("user blocked")

# get blocked user
@app.route('/get_blocked_user', methods=['GET'])
def get_blocked_user(): 
    temp_list = list(collection_blocked.find({}))
    for item in temp_list:
      item['_id'] = str(item['_id'])
    return jsonify(temp_list)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename.endswith('.mp3'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER_AUDIO'], uploaded_file.filename)
        uploaded_file.save(file_path)
        new,_ =uploaded_file.filename.rsplit('.', 1)
        new_name=new+".wav"
        print(new_name)
        des_path = os.path.join(app.config['UPLOAD_FOLDER_AUDIO'], new_name)
        run_audio_script(file_path, des_path)
        with open('Human_generated_Logs/data/input_data/new_audio.txt', 'r') as file:
          conversation_lines = file.readlines()

        conversation_text = ''.join(conversation_lines)

        shutil.rmtree("Human_generated_Logs/data/audio/")
        shutil.rmtree("Human_generated_Logs/data/input_data/")
        os.mkdir("Human_generated_Logs/data/audio/")
        os.mkdir("Human_generated_Logs/data/input_data/")
        
        print(conversation_text)
        return jsonify(conversation_text)
    else:
      file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
      uploaded_file.save(file_path)
      run_another_script()
      uploadtoDB()
      try:
        pipeline = [
          {"$sample": {"size": 1}}
        ]
        result_list = list(collection_input.aggregate(pipeline))
        random_instance = result_list[0]
        random_instance['_id'] = str(random_instance['_id'])
        logging.info('random_instance')    
        return jsonify(random_instance)
      except Exception as e:
        return jsonify({"error": str(e)}), 500   
    
def uploadtoDB():
  data_directory = 'database_push/'
  if collection_input.count_documents({}) > 0:
        collection_input.delete_many({})
        print("Collection emptied.")
  for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_directory, filename)
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                json_data = [row for row in csv_reader]

                if json_data:
                    collection_input.insert_many(json_data)
                    print(f"Inserted {len(json_data)} documents from {filename} into MongoDB.")
                else:
                    print(f"No data in {filename}")
         

def run_another_script():
    script_path = "System_generated_Logs/scripts/log_file_input.py"
    try:
        subprocess.run(["python", script_path], check=True)
        print("Other script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing the other script: {e}")

def run_audio_script(source_path, des_path):
    subprocess.run(['python', 'Human_generated_Logs/scripts/audio_to_txt.py', source_path, des_path])
        
        
@app.route('/api/upload/rules', methods=['POST'])
def upload_rule_file():
    uploaded_rule_file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER_RULES'], uploaded_rule_file.filename)
    uploaded_rule_file.save(file_path)
    
    return jsonify({'message': f'Rule file {uploaded_rule_file.filename} uploaded successfully'})

@app.route('/get_rules', methods=['GET'])
def get_rules():
    directory_path = app.config['UPLOAD_FOLDER_RULES']
    file_list = os.listdir(directory_path)
    rule_texts = []

    for filename in file_list:
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r') as f:
            rule_text = f.read()
            rule_texts.append(rule_text)
        os.remove(file_path)

    # Remove the directory and its contents after reading the files
    # shutil.rmtree(directory_path)


    return jsonify({'rule_texts': rule_texts})

@app.route('/input_random_instance', methods=['GET'])
def input_random_instance():
    try:
        pipeline = [
          {"$sample": {"size": 1}}
        ]
        result_list = list(collection_input.aggregate(pipeline))
        random_instance = result_list[0]
        random_instance['_id'] = str(random_instance['_id'])
        logging.info('random_instance')    
        return jsonify(random_instance)
    except Exception as e:
      return jsonify({"error": str(e)}), 500
    
  
# For Customer
# step 1 - fetch customer-cr details 
@app.route('/customer_random_instance') 
def customer_random_instace(): 
  try:
      pipeline = [
        {"$sample": {"size": 1}}
      ]
      result_list = list(collection_customer.aggregate(pipeline))
      random_instance = result_list[0]
      random_instance['_id'] = str(random_instance['_id'])
      return jsonify(random_instance)
  except Exception as e: 
    return jsonify({"error in customer random instance": str(e)}), 500  

# testing
@app.route('/test', methods=['POST'])
def test():
    temp_data = {
        "answer": " I'm just an AI, I don't have access to external information or systems, so I can't provide you with the exact password policy for Flipkart. Additionally, it is not appropriate or ethical to share or use someone else's password policies without proper authorization. It is important to respect the security and privacy of others' systems and data. If you have any other questions or concerns, feel free to ask!"
    }
    return jsonify(temp_data)

if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    app.run(debug=True)