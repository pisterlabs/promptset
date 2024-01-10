import io
import json
import numpy as np
import os
import transformers
import pydantic
import uuid

from blank_to_bard import llm_explainability_prompt
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from lime import lime_text
from pydantic import BaseModel

load_dotenv()  # take environment variables from .env.

_BUCKET_NAME = 'blank-to-bard'


class LIMERequest(pydantic.BaseModel):
	text: str
	probs: List[float]

class LLMRequest(pydantic.BaseModel):
	text: str

app = FastAPI()

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
# Define a root `/` endpoint that shows all the available endpoints
async def root():
    return {"message": "Welcome to the blank-to-bard explainability API!"}

@app.get("/health")
def health_check():
    return {"status": "Healthy"}

@app.post("/explainability/lime")
async def explain_lime(request: LIMERequest):
	print(request.text)
	print(request.probs)

	class_names = ['blank', 'bard']
	if len(request.probs) != 2 or not request.text:
		return dict()

	def prediction_probs(_: str):
		return np.array(request.probs)

	explainer = lime_text.LimeTextExplainer(class_names=class_names)
	explanation = explainer.explain_instance(request.text, prediction_probs)
	
	weightage = explanation.as_list()
	
	fig = explanation.as_pyplot_figure()
	buf = io.BytesIO()
	fig.savefig(buf, format='png')

	blob_id = str(uuid.uuid4())
	img_path = f'gs://{_BUCKET_NAME}/{blob_id}/lime.png'
	_save_image_to_gcs(buf, _BUCKET_NAME, blob_id)

	return {'weightage': weightage, 'img_path': img_path}

def _save_image_to_gcs(buf: io.BytesIO, bucket_name: str, blob_id: str):
	client = storage.Client()
	bucket = client.get_bucket(bucket_name)
	blob = bucket.blob(f'{blob_id}/lime.png')
	blob.upload_from_file(buf, content_type='image/png', rewind=True)

@app.post("/explainability/llm")
async def explain_llm(request: LLMRequest):
	llm = OpenAI(temperature=0.7, model_name="gpt-4")

	question = request.text

	prompt_template = llm_explainability_prompt.PROMPT_PREFIX + question + llm_explainability_prompt.PROMPT_SUFFIX
	prompt = PromptTemplate(template=prompt_template, input_variables=[])

	chain = LLMChain(prompt=prompt,llm=llm)

	pred = chain.predict()

	try:
		return json.loads(pred)
	except ValueError:
		return dict{}
