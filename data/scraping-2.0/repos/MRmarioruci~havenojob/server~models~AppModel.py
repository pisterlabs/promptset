from utils import log, getStandardResponse, getResponseDict
from flask import jsonify
from dotenv import load_dotenv
from sqlalchemy import text
import json
import os
import openai
import tiktoken
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class AppModel:
	def num_tokens_from_string(string: str, encoding_name: str) -> int:
		encoding = tiktoken.get_encoding(encoding_name)
		num_tokens = len(encoding.encode(string))
		return num_tokens

	def promptChatGPT(self, messages, temperature):
		try:
			#message_tokens = self.num_tokens_from_string('testing', 'gpt-3.5-turbo')
			#print(message_tokens);
			# Set the desired maximum token limit
			max_tokens = 2048  # Adjust as per your model's limit

			# Check if the total token count exceeds the limit
			#if message_tokens > max_tokens:
			#	raise ValueError("Total token count exceeds the maximum limit.")

			response = openai.ChatCompletion.create(
				model=os.getenv('MODEL_VERSION'),
				messages=messages,
				temperature=temperature if temperature else 0.1,
				max_tokens=max_tokens
			)['choices'][0]['message']['content']

			return response
		except ValueError as e:
			log('ERROR', 'ValueError: {}'.format(str(e)))
			return 'Error: {}'.format(str(e))
		except:
			log('ERROR','An error occurred on chatcompletion')
			return 'An error occured on chatcompletion'


	def getJobMatch(self, candidate_job_title:str, candidate_profile:str, job_title:str, job_description:str, summarize_profile:bool, summarize_job:bool):
		response = getResponseDict()
	
		#criteria = self.generateCriteriaBasedOnJobTitle(job_title)
		criteria = [
			'Field Relevance',
			'Education and qualifications',
			'Work experience',
			'Skills and competencies',
			'Cultural fit',
			'Achievements and accomplishments',
			'Problem-solving and critical thinking',
			'Communication and interpersonal skills',
			'Adaptability and learning agility',
		]
		profileSummary = candidate_profile
		if summarize_profile:
			profileSummary = self.generateProfileSummary(candidate_job_title, candidate_profile)
			profileSummary = profileSummary['data'] if profileSummary['status'] == 'ok' else candidate_profile

		jobSummary = job_description
		if summarize_job:
			jobSummary = self.generateJobDescriptionSummary(job_title, job_description)
			jobSummary = jobSummary['data'] if jobSummary['status'] == 'ok' else job_description

		print('Generating evaluation...')
		evaluation = self.promptChatGPT([
			{"role": "system", "content": "Forget"},
			{"role": "system", "content": "You are a job matching evaluator."},
			{"role": "system", "content": "Evaluate the match between the candidate current job title-cv and the given job title - job description."},
			{"role": "system", "content": "Provide a percentage range as your evaluation."},
			{"role": "system", "content": 'Format your response like the following example in json format: {"percentageLow": "Number", "percentageHigh": "Number"}'},
			{"role": "system", "content": f"You Evaluate with these criteria in mind: {criteria}"},
			{"role": "system", "content": "You are a strict and objective evaluator"},
			{"role": "user", "content": f"Candidate's current job title: {candidate_job_title}."},
			{"role": "user", "content": f"Candidate's profile summary: {profileSummary}."},
			{"role": "user", "content": f"Job title: {job_title}."},
			{"role": "user", "content": f"Job description: {jobSummary}."},
			{"role": "user", "content": "Respond with the percentage range, nothing more."}
		], 0.1)
		if evaluation:
			response['data'] = json.loads(evaluation)

		if response['data']: response['status'] = 'ok'
		return response
	
	def generateProfileSummary(self, candidate_job_title:str, candidate_profile:str):
		print('Generating profile summary')
		response = getResponseDict()
		response['data'] = self.promptChatGPT([
			{"role": "system", "content": "You are a job candidates profile summarizer"},
			{"role": "system", "content": "You will receive raw text data from a cv and organize it, create a structured summary with only the necessary information."},
			{"role": "system", "content": "An interviwer must be able to skim through it quickly but get the most relevant information as well. No important information loss"},
			{"role": "system", "content": "The format should be something like; Current Job Title: Software Engineer \n Years of experience: 6 etc."},
			{"role": "user", "content": f'candidate_job_title: {candidate_job_title.strip()}'},
			{"role": "user", "content": f'candidate_cv: {candidate_profile.strip()}'},
			{"role": "user", "content": f'your response should only be the summary, nothing more'},
			{"role": "user", "content": f'Format the response as html for better readability. Use <div> and <b> no h tags'},
		], 0.1)
		if response['data']: response['status'] = 'ok'
		return response
	
	def generateJobDescriptionSummary(self, job_title:str, job_description:str):
		print('Generating job description summary')
		response = getResponseDict()
		keypoints = [
			"Job Title",
			"Job Summary/Objective",
			"Key Responsibilities",
			"Qualifications and Requirements",
			"Company Overview",
			"Benefits and Perks",
			"Location and Work Schedule",
			"Career Development Opportunities",
			"Application Process"
		]
		response['data'] = self.promptChatGPT([
			{"role": "system", "content": "You are a job description summarizer"},
			{"role": "system", "content": "You will receive raw text data from a job description"},
			{"role": "system", "content": "You will provide a structured summary of the job description. Only necessary information"},
			{"role": "system", "content": f"You will have to detect these keypoints:{keypoints}"},
			{"role": "user", "content": f'job_title: {job_title.strip()}'},
			{"role": "user", "content": f'job_description: {job_description.strip()}'},
			{"role": "user", "content": f'your response should only be the summary, nothing more'},
			{"role": "user", "content": f'Format the response as html for better readability. Use <div> and <b> no h tags'},
		], 0.1)
		if response['data']: response['status'] = 'ok'
		return response
	
	def generateCriteriaBasedOnJobTitle(self, job_title:str):
		response = getResponseDict()
		response['data'] = self.promptChatGPT([
			{"role": "system", "content": "You are a candidate evaluation criteria generator"},
			{"role": "system", "content": "You will receive one argument: job_title. Based on that job title you have to generate at least 5 most relevant criteria for evaluating the candidate "},
			{"role": "system", "content": "You should generate relevant criteria to the job title. Example: job_title: Software Engineer -> Criteria = [Relevant experience, Skills & Qualifications, Technical proficiency]"},
			{"role": "system", "content": "Your response should be in a table format like [criteria 1, criteria 2]"},
			{"role": "system", "content": "You will only return the result array and nothing more. No other text from you."},
			{"role": "system", "content": "On every request you will forget about the previous job and response you gave me"},
			{"role": "user", "content": f'job_title: {job_title.strip()}'},
			{"role": "user", "content": f'your response should only be the array, nothing more'},
		], 0.1)
		if response['data']: response['status'] = 'ok'
		return response
	
	def preregister(self, email:str, SessionLocal):
		response = getResponseDict()
		with SessionLocal() as session:
			try:
				q = 'INSERT INTO `Emails`(`email`, `creationDate`) VALUES(:email, NOW())'
				result = session.execute(text(q), {
					"email": email
				})
				session.commit()
				if result.lastrowid:
					response['status'] = 'ok'
					response['data'] = True

			except Exception as e:
				response['data'] = True
				response['status'] = 'ok'
				log('ERROR', e)

		return response

	def generateCoverLetter(self, job_title:str, company:str, candidate_profile: str, extraInstructions:str):
		print('Generating job description summary')
		response = getResponseDict()
		response['data'] = self.promptChatGPT([
			{"role": "system", "content": "Forget. You are a cover letter writer"},
			{"role": "system", "content": "You will receive a job_title, a company_name and a candidate_profile"},
			{"role": "system", "content": "You will provide a relevant cover letter matching the candidate_profile to the role"},
			{"role": "system", "content": "If there is no candidate_profile then make the cover letter generic"},
			{"role": "user", "content": f'job_title: {job_title.strip()}'},
			{"role": "user", "content": f'company_name: {company.strip()}'},
			{"role": "user", "content": f'candidate_profile: {candidate_profile.strip()}'},
			{"role": "user", "content": f'If any extra instructions are provided, follow them. Extra instructions: {extraInstructions}'},
			{"role": "user", "content": f'your response should only be the cover, nothing more, in an html format with 2 <br> for new lines. Nicely formatted'},
		], 0.2)
		if response['data']: response['status'] = 'ok'
		return response