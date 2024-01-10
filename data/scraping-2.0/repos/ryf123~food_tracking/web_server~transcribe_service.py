import ast
import os
import openai
import re
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from db_access import DBAccess

class TranscribeService:
	def __init__(self):
		openai.api_key = os.getenv("OPENAI_API_KEY")

	def summarize_calorie_intake(self, text):
		llm = OpenAI(temperature=0)

		estimate_calorie_template = """Estimate the calorie for each item, and put them in a json array in the format of [{{"name": food_name, "amount": food_amount, "calorie": estimated_calorie}}]: {text} """
		estimate_calorie_prompt_template = PromptTemplate(input_variables=["text"], template=estimate_calorie_template)
		estimate_calorie_chain = LLMChain(llm=llm, prompt=estimate_calorie_prompt_template, output_key="table")

		overall_chain = SequentialChain(chains=[estimate_calorie_chain], input_variables=["text"], output_variables=["table"], verbose=True)
		result = overall_chain({'text': text})
		print(result)
		processed_result = self.process_transcribed_result(result['table'])
		return processed_result

	def nutrition_analysis(self, user_id):
		llm = OpenAI(temperature=0)
		db_access = DBAccess()
		records = db_access.get_food_records_by_date(user_id)
		food_analysis_template = """You are playing the role of a nutritionist now, and provide specific guidance based on my food records: {records}"""
		estimate_calorie_prompt_template = PromptTemplate(input_variables=["records"], template=food_analysis_template)
		estimate_calorie_chain = LLMChain(llm=llm, prompt=estimate_calorie_prompt_template, output_key="analysis")

		overall_chain = SequentialChain(chains=[estimate_calorie_chain], input_variables=["records"], output_variables=["analysis"], verbose=True)
		result = overall_chain({'records': records})
		print(result)
		return result

	def process_transcribed_result(self, results):
		data_list = ast.literal_eval(results)
		total_calorie = 0
		for item in data_list:
			total_calorie += int(item['calorie'])
		return {'total_calorie': total_calorie}

	def transcribe_audio(self, audio_content):
		# Transcribe the audio using OpenAI Whisper API
		# Replace this with your actual transcription logic
		try:
			transcript = openai.Audio.transcribe("whisper-1", audio_content, prompt="The input is related to food consumed on a specific day")
			return transcript.text
		except Exception as e:
			print("An error occurred:", str(e))

	def transcribe_audio_file(self, file_name):
		# Transcribe the audio using OpenAI Whisper API
		# Replace this with your actual transcription logic
		try:
			audio_file= open(file_name, "rb")
			transcript = openai.Audio.transcribe("whisper-1", audio_file, prompt="The input is related to food consumed on a specific day")
			return transcript.text
		except Exception as e:
			print("An error occurred:", str(e))
