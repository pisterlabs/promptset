from sarif import loader
import json

import attrs
import sarif_om

import json
import os
import sys

import boto3
import botocore

# module_path = ".."
# sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

def get_sarif_class(schema_ref):
    class_name = schema_ref.split('/')[-1]
    class_name = class_name[0].capitalize() + class_name[1:]
    return getattr(sarif_om, class_name)


def get_field_name(schema_property_name, cls):
    for field in attrs.fields(cls):
        if field.metadata.get('schema_property_name') == schema_property_name:
            return field.name
    return schema_property_name


def get_schema_properties(schema, schema_ref):
    cursor = schema
    for part in schema_ref.split('/'):
        if part == '#':
            cursor = schema
        else:
            cursor = cursor[part]
    return cursor['properties']


def materialize(data, cls, schema, schema_ref):
    fields = {}
    extras = {}
    props = get_schema_properties(schema, schema_ref)

    for key, value in data.items():
        field_name = get_field_name(key, cls)

        if key not in props:
            extras[field_name] = value
            continue

        if '$ref' in props[key]:
            schema_ref = props[key]['$ref']
            field_cls = get_sarif_class(schema_ref)
            fields[field_name] = materialize(value, field_cls, schema, schema_ref)

        elif 'items' in props[key]:
            schema_ref = props[key]['items'].get('$ref')
            if schema_ref:
                field_cls = get_sarif_class(schema_ref)
                fields[field_name] = [materialize(v, field_cls, schema, schema_ref) for v in value]
            else:
                fields[field_name] = value
        else:
            fields[field_name] = value

    obj = cls(**fields)
    obj.__dict__.update(extras)
    return obj


path_to_sarif_file = "spotbugs-sarif.json"

sarif_data = loader.load_sarif_file(path_to_sarif_file)
issue_count_by_severity = sarif_data.get_result_count_by_severity()
error_histogram = sarif_data.get_issue_code_histogram("error")
warning_histogram = sarif_data.get_issue_code_histogram("warning")
note_histogram = sarif_data.get_issue_code_histogram("note")

print(f"Issue count by severity: {issue_count_by_severity}")
print(f"Error histogram: {error_histogram}")
print(f"Warning histogram: {warning_histogram}")
print(f"Note histogram: {note_histogram}")

with open('spotbugs-sarif.json', 'r') as file:
    data = json.load(file)

with open('sarif-schema-2.1.0.json', 'r') as file:
    schema = json.load(file)

sarif_log = materialize(data, sarif_om.SarifLog, schema, '#')
print(sarif_log)

print("=== SARIF runs (typically only one) ===")
sarif_runs = sarif_log.runs
run = sarif_runs[0]
print(run)

print("=== Rules ===")
rules = run.tool.driver.rules
print(rules)

print("=== Results ===")
results = run.results
print(results)

# Evaluate Bedrock invocation against the first SARIF result.
print("=== Result[0] ===")
# result0 = results[0]
# 12 -> Command Injection in example.
result0 = results[12]
print(result0)

result0_message_text = result0.message.text
result0_rule_id = result0.rule_id
result0_rule_index = result0.rule_index
result0_rule = rules[result0_rule_index]
result0_rule_target_name = result0_rule.relationships[0].target.tool_component.name
result0_rule_target_id = result0_rule.relationships[0].target.id
# Consider the first location for now.
result0_location0 = result0.locations[0]
result0_artifact_location_uri = result0_location0.physical_location.artifact_location.uri
result0_region_start_line = result0_location0.physical_location.region.start_line

# Assume we have Maven source.
file_path = f"src/main/java/{result0_artifact_location_uri}"

print("=== File location ===")
print(result0_artifact_location_uri)

# Extract code snippet 10 lines before and after around the "result0_region_start_line" from the file designated by  result0_artifact_location_uri
# Declare string variable with name code_snippet.
# code_snippet = ""
print("=== File content ===")
with open(file_path, 'r') as file:
    lines = file.readlines()
    start_line = result0_region_start_line
    end_line = start_line + 10
    start_line = max(0, start_line - 10)
    end_line = min(len(lines), end_line)
    code_snippet = "".join(lines[start_line:end_line])
    print(code_snippet)


# TODO: Extract some code snippets from the physical locations and files in "src" folder.

# code_snippet = """
#         addActionError(e.getMessage());
# 		        return ERROR;
# 	        }
#         }
#
# 	    // Throwing Exceptions for RTAL
# 	    try{
# 		    byte[] theKey={'a'};
# 		    SecretKeySpec skeySpec = new SecretKeySpec(theKey, "AES");
# 		    Cipher cipher = Cipher.getInstance("AES");
# 		    cipher.init(Cipher.ENCRYPT_MODE, skeySpec); // Exception thrown:java.security.InvalidKeyException: Invalid AES key length: 1 bytes
# 	    }
# 	    catch (Exception e){
# 	    }
#
# 	    try{
# 		    SecretKeySpec skeySpec = new SecretKeySpec(from.getBytes(), "AES/CBC/PKCS7Padding");
# 		    Cipher cipher = Cipher.getInstance("AES/CBC/PKCS7Padding");
# 		    cipher.init(Cipher.ENCRYPT_MODE, skeySpec); // Exception thrown:java.security.NoSuchAlgorithmException: Cannot find any provider supporting AES/CBC/PKCS7Padding
# 	    }
# """

# prompt = """
# SAST tool has identified the following Java code has security vulnerability "{}-{}: {}":
# ---
# {}
# ---
# Do you think this code really is vulnerable?
# And if it is vulnerable, tell me how to fix it and suggest fixed code.
# Respond in Korean.
# """.format(result0_rule_target_name, result0_rule_target_id, result0_message_text, code_snippet)

prompt = """
SAST 툴이 아래 자바 코드가 "{}-{}: {}" 취약점을 가지고 있다고 진단하였습니다:
---
{}
---
이 코드가 실제로 취약한가요?
만약 그렇다면 조치할 수 있는 방법고 조치된 코드를 제시해 주세요.
답변은 한국어로 해주세요.
""".format(result0_rule_target_name, result0_rule_target_id, result0_message_text, code_snippet)

print(prompt)

# ---- ⚠️ Un-comment and edit the below lines as needed for your AWS setup ⚠️ ----

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  # E.g. "us-east-1"
# os.environ["AWS_PROFILE"] = "<YOUR_PROFILE>"
# os.environ["BEDROCK_ASSUME_ROLE"] = "<YOUR_ROLE_ARN>"  # E.g. "arn:aws:..."


boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

# body = json.dumps({
#     "inputText": prompt,
#     "textGenerationConfig": {
#         "maxTokenCount": 4096,
#         "stopSequences": [],
#         "temperature": 0,
#         "topP": 0.9
#     }
# })

# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html
# https://docs.aws.amazon.com/bedrock/latest/userguide/api-methods-run-inference.html
# {
#     "prompt": "\n\nHuman:<prompt>\n\nAssistant:",
#     "temperature": float,
#     "top_p": float,
#     "top_k": int,
#     "max_tokens_to_sample": int,
#     "stop_sequences": ["\n\nHuman:"]
# }
body = json.dumps({
    "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
    "max_tokens_to_sample": 1024,
    "temperature": 0.05,
    "top_p": 0.9,
    "stop_sequences": ["\n\nHuman:"]
})

# https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html
modelId = 'anthropic.claude-v2:1'
# modelId = 'anthropic.claude-instant-v1'
accept = 'application/json'
contentType = 'application/json'
outputText = "\n"
try:
    response = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept=accept,
                                          contentType=contentType)
    response_body = json.loads(response.get('body').read())
    # outputText = responseBody.get('results')[0].get('outputText')
    # text
    print_ww(response_body.get('completion'))
except botocore.exceptions.ClientError as error:
    if error.response['Error']['Code'] == 'AccessDeniedException':
        print(f"\x1b[41m{error.response['Error']['Message']}\
                \n해당 이슈를 트러블슈팅하기 위해서는 다음 문서를 참고하세요.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")

    else:
        raise error

# The relevant portion of the response begins after the first newline character.
# Below we print the response beginning after the first occurence of '\n'.

# remediation = outputText[outputText.index('\n') + 1:]
# print_ww(remediation)


# import openai
#
# # Set the OpenAI API key
# api_key = 'your_api_key'
# openai.api_key = api_key
#
# # Define the prompt for the language model
# prompt = "Create a short story about an astronaut who discovers a new planet."
#
# # Set the model name and parameters
# model = "text-davinci-003"
# temperature = 0.7
# max_tokens = 150
#
# # Generate the completion using the prompt and model
# response = openai.Completion.create(
#     engine=model,
#     prompt=prompt,
#     temperature=temperature,
#     max_tokens=max_tokens
# )
#
# # Print the generated completion
# print(response.choices[0].text.strip())

#
# Now try to get inferences for all the finding.
#
