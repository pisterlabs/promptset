from google.cloud import storage
from diagrams import Diagram
import ast
from PIL import Image
from io import BytesIO
import boto3
import subprocess
import tempfile
from langchain.schema import (HumanMessage)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
import os
os.environ["PATH"] += os.pathsep + "/usr/local/opt/graphviz/bin"


chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)


def return_infra_code(prompt):
    template = "I want to create an architecture diagram using diagrams library in python, of the service whose description is as follows: ""{content}"". Generate a string of code to make the diagram in python. Just return ONLY the python code as a STRING in your answer response and no other data AT ALL. sample response: ""from diagrams import Diagram, Cluster, Edge, Node"". "
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    # example_human = HumanMessagePromptTemplate.from_template("Hi")
    # example_ai = AIMessagePromptTemplate.from_template("Argh me mateys")
    human_template = "Strictly return only the Python code in string format and no other extra string data"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(
        {'text': prompt, 'content': 'Strictly return only the Python code in string format and no other extra string data'})
    return result

    # template=
    # system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    # human_template="I like making systems diagrams using `diagrams` package in python"
    # human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    # chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # return chat([HumanMessage(chat_prompt.format_prompt(content=prompt).to_messages())])


# Initialize the Google Cloud Storage client
# storage_client = storage.Client()

GOOGLE_APPLICATION_CREDENTIALS = "/Users/ankitsanghvi/Desktop/cloudpilot-be/credentials.json"


def generate_diagram_image(code: str, image_format: str = "png") -> BytesIO:
    # Parse the code and execute it to generate the diagram
    code_ast = ast.parse(code)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change the current working directory to the temporary directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        # Execute the code to generate the diagram
        exec(compile(code_ast, filename="<ast>", mode="exec"), globals())

        # Get the generated diagram filename
        diagram_filename = None
        for filename in os.listdir(temp_dir):
            if filename.endswith("." + image_format):
                diagram_filename = os.path.join(temp_dir, filename)
                break

        # Save the generated diagram to a BytesIO buffer
        buffer = BytesIO()
        Image.open(diagram_filename).save(buffer, image_format)

        # Change the current working directory back to the original
        os.chdir(original_cwd)

    buffer.seek(0)
    return buffer


def upload_image_to_s3(buffer: BytesIO, key: str, bucket_name: str, image_format: str = "png") -> str:
    s3 = boto3.client('s3')
    s3.upload_fileobj(buffer, bucket_name, key, ExtraArgs={
                      "ContentType": f"image/{image_format}"})

    # Generate the S3 URL
    return f"https://{bucket_name}.s3.amazonaws.com/{key}"


def upload_image_to_gcs(buffer: BytesIO, key: str, bucket_name: str, image_format: str = "png") -> str:
    # Create a Google Cloud Storage client
    client = storage.Client()

    # Get the bucket from the client
    bucket = client.get_bucket(bucket_name)

    # Create a blob (i.e., an object) with the given key
    blob = bucket.blob(key)

    # Set the content type of the blob based on the image format
    blob.content_type = f"image/{image_format}"

    # Upload the image buffer to the blob
    blob.upload_from_string(buffer.getvalue(), content_type=blob.content_type)

    # Generate the Google Cloud Storage URL
    return f"https://storage.googleapis.com/{bucket_name}/{key}"


def remove_unwanted_lines(code: str) -> str:
    lines = code.split("\n")
    clean_lines = [line for line in lines if not line.startswith("Here's")]
    return "\n".join(clean_lines)


def generate_diagram_image_and_upload_to_s3(code: str, bucket_name: str, image_format: str = "png") -> str:
    # Generate a temporary image from the code
    stripped_code = remove_unwanted_lines(code[1:-1].replace('\\n', '\n'))
    image_buffer = generate_diagram_image(stripped_code, image_format)

    # Generate a unique key for the image
    key = f"diagrams/{os.urandom(8).hex()}.{image_format}"

    # Upload the image to S3 and get the URL
    url = upload_image_to_s3(image_buffer, key, bucket_name, image_format)
    # url = upload_image_to_gcs(image_buffer, key, "cloudpilot-systems-diagrams-bucket", image_format)

    return url

# def generate_diagram_image_and_upload_to_s3(code: str, object_key: str):
#     # Create a temporary file for the diagram
#     # fd, temp_filename = tempfile.mkstemp(suffix=".png")
#     # os.close(fd)

#     try:
#         # Execute the code_str to generate the diagram and save it to the temporary file
#         with Diagram("temp_filename", show=False, outformat="png",filename="/Users/ankitsanghvi/Desktop/cloudpilot-be/my_diagram"):
#             exec(code)
#             # diagram.render()

#         # Create an S3 client and upload the temporary file to the specified bucket
#         s3 = boto3.client("s3", region_name='ap-south-1')
#         key = f"diagrams/{object_key}.png"
#         s3.upload_file("/Users/ankitsanghvi/Desktop/cloudpilot-be/my_diagram.png", "cloudpilot-systems-design-diagrams", key)

#         # Return the URL of the uploaded file in the S3 bucket
#         url = f"https://cloudpilot-systems-design-diagrams.s3.amazonaws.com/{key}"

#     finally:
#         # Delete the temporary file
#         os.remove(temp_filename)

#     return url
