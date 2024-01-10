from langchain.chat_models import ChatOpenAI
import boto3
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from graphviz import Digraph

from dotenv import load_dotenv

FILE_NAME = "conceptmap"

chat = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0)
load_dotenv()
def get_concept_map_code(prompt):
    template="For the following excerpt, generate code template that the `graphviz` library of python can process to make a concept map: ""{text}"". You are to answer the question in the following format: ""{content}"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="""Don't return any explanation or supporting text. I want you to ONLY return the appropriate and exact "graphviz template code" for this map as your response"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result= chain.run({'text': prompt, 'content': """Don't return any explanation or supporting text. I want you to ONLY return the appropriate and exact "graphviz template code" for this map as your response"""})
    return result

def generate_diagram_image(openai_output_code):
    dot = Digraph('ConceptGraph', format='png')
    exec(openai_output_code)
    print(dot.source)
    dot.render(FILE_NAME, view=True)

def generate_concept_map_image_and_upload_to_s3(code, file_name, bucket_name):
    # Create an S3 client
    generate_diagram_image(code)
    s3 = boto3.client('s3')
    file_name= FILE_NAME + ".png"

    # Uploads the given file using a managed uploader, which will split up the
    # file if it's large and uploads parts in parallel.
    s3.upload_file(file_name, bucket_name, file_name)

    # Generate the URL for the uploaded file
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/concept-maps/{file_name}"

    # Delete the file from local after uploading
    os.remove(file_name)

    return s3_url