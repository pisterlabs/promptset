import re, json
import openai

from logger import log as logger
from logger import ansicodes

class answeringEngine:
	def __init__(self, openAiKey, user):
		openai.api_key = openAiKey
		self.presetQuestionResponses = self.loadPresetResponses(user)
		return


	def loadPresetResponses(self, user):
		return {
			'usersName': {
				'keyWords': ['your name', 'preferred name'],
				'response': {
					"string": user.fullName
				}
			},
			'usersPronouns': {
				'keyWords': ['pronoun'],
				'response': {
					"string": user.pronouns
				}
			},
			'usersGit': {
				'keyWords': ['git', 'bitbucket'],
				'response': {
					"string": user.gitURL
				}
			},
			'usersLinkedIn': {
				'keyWords': ['linkedin'],
				'response': {
					"string": user.linkedIn
				}
			},
			'usersWebsite': {
				'keyWords': ['website', 'portfolio'],
				'response': {
					"string": user.website
				}
			},
			'rightToWork': {
				'keyWords': ['visa', 'right to work'],
				'response': {
					"bool": user.canWork
				}
			},
			'salary': {
				'keyWords': ['salary'],
				'response': {
					"string": str(user.salary) + " " + user.salaryCurrency,
					"int": int(user.salary)
				}
			},
			'location': {
				'keyWords': ['country', 'locat', 'based', 'live'],
				'response': {
					"string": user.location
				}
			},
			'notice': {
				'keyWords': ['notice'],
				'response': {
					"string": "I can start work immediately.",
					"int": 0
				}
			},
			'criminal': {
				'keyWords': ['criminal'],
				'response': {
					"string": "No, I have never been convicted of a criminal offence.",
					"bool": False
				}
			}
		}


	def askGPT(self, primedString, tokens=256, temp=1.1):
		print("Generating GPT Response...")
		response = openai.Completion.create(
		  model="text-davinci-002",
		  prompt=primedString,
		  max_tokens=tokens,
		  temperature=temp
		)
		return response['choices'][0]['text'].lstrip('\n')

	# Question Types;
	# 	String*
	# 	Int*
	# 	Bool*
	#	Multiple Choice
	#	Date
	def answerQuestion(self, question, qresponsetype, questionBackground = "", forceGPT = False, choices = None):
		# Classify Question to see if it has preset answers
		questionType = self.classifyQuestion(question)

		# Check responseType Set
		if qresponsetype == None:
			return None

		# Check GPT is enabled, it has a canned response, and the canned response is the correct type
		if (not forceGPT) and questionType and (qresponsetype.lower() in self.presetQuestionResponses[questionType]['response']):
			# Use the existing canned response, or call the canned response function
			questionResponse = self.presetQuestionResponses[questionType]['response'][qresponsetype]
			if callable(questionResponse):
				questionResponse = questionResponse(question, qresponsetype)
		# Otherwise Get GPT to respond
		else:
			primedQuestion = self.primeQuestionForGPT(question, qresponsetype, choices)
			try:
				questionResponse = self.askGPT( questionBackground + primedQuestion )
				logger.debug("[RawResponse]: %s" % questionResponse)
				#Return the question response in the correct format.
				questionResponse = self.castResponse( questionResponse, qresponsetype, choices )
				#Todo: try and catch bad responses

			except:
				print(ansicodes.GREEN + "GPT3 was unable to respond" + ansicodes.RST)
				questionResponse = None

		return questionResponse;

	def classifyQuestion(self, question):
		# Future Todo: add weighting to keyWords: e.g
		# Why do you want to work at github? - Should not match, "why" would have a negative weight.
		# Please describe a recent project you've worked on, feel free to link a git repo. - Should match projects, "Project" would be weighed higher than "git" and "git" would be a keyword in projects

		threshold = 0.1
		pQuestionTypes = []

		for presetQuestion in self.presetQuestionResponses:
			keywordMatches = 0
			for word in self.presetQuestionResponses[presetQuestion]["keyWords"]:
				if word.lower() in question.lower():
					keywordMatches += 1 / len(self.presetQuestionResponses[presetQuestion]["keyWords"])
			pQuestionTypes.append( (presetQuestion, keywordMatches) )

		bestMatch = max( pQuestionTypes, key = lambda k: k[1] )

		return bestMatch[0] if bestMatch[1] > threshold else None

	# Primes all 
	def primeQuestionForGPT(self, question, qtype, choices = None):
		qtype = qtype.lower()
		if qtype == "multiple choice":
			choices = "\n".join(map(lambda x: "%d. %s" % (x[0]+1, x[1]), enumerate(choices)))
			print(choices)
			return "\nQ: %s (Type the number for the correct response)\n%s\n" % (question, choices)
		elif qtype == "bool":
			return "\nQ: %s (Y/N)\nA:" % (question)
		elif qtype == "int":
			return "\nQ: %s (number)\nA:" % (question)
		else:
			return "\nQ: %s\nA: " % (question)

	def castResponse(self, questionResponse, qresponsetype, choices = None):
		qresponsetype = qresponsetype.lower();
		if qresponsetype == "string":
			#Respond With The Whole GPT Response
			return questionResponse
		if qresponsetype == "int":
			#Respond With The First Number GPT Provides, Or None
			m = re.search(r'\d+', questionResponse)
			return int(m.group(0)) if m else None
		if qresponsetype == "bool":
			#Check for n or no in the Response
			questionResponse = questionResponse.lower()
			n = re.search(r'(^|[^\w])no?([^\w]|$)', questionResponse)
			y = re.search(r'(^|[^\w])y(es)?([^\w]|$)', questionResponse)
			return bool(y) if ( y or n ) else None
		if qresponsetype == "multiple choice":
			#Get the Number from the response
			m = re.search(r'\d+', questionResponse)
			return int(m.group(0)) if int(m.group(0)) <= len(choices) else None
class user:

	countryAliases: {
		"UK": ["UK", "GB", "Scotland", "England", "United Kingdom", "Britian"],
		"US": ["US" "USA", "United States", "America", "The United States of America"]
	}

	def __init__(self, userdata):
		self.fullName = userdata["fullName"]
		self.email = userdata["email"]
		self.pronouns = userdata["pronouns"]
		self.gitURL = userdata["gitURL"]
		self.linkedIn = userdata["linkedIn"]
		self.website = userdata["website"]
		self.telephone = userdata["telephone"]
		self.salary = userdata["salary"]
		self.salaryCurrency = userdata["salaryCurrency"]
		self.countries = userdata["rightToWorkCountries"]
		self.location = userdata["location"]
		return

	def canWork(self, question, qresponsetype):
		canWorkInCountry = False
		for country in self.countries:
			for alias in country:
				# May Need Better Matching
				if alias in question:
					canWorkInCountry = True

		if qresponsetype == "bool":
			return canWorkInCountry
		elif qresponsetype == "string":
			return "Yes, I can work without a visa." if canWorkInCountry else "No, I would require visa sponsorship."

		raise Exception("Question Response Type Is Invalid For This Question")