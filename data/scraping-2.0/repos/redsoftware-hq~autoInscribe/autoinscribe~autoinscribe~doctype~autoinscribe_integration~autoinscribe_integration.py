# Copyright (c) 2024, RedSoft Solutions Pvt. Ltd. and contributors
# For license information, please see license.txt

import frappe
from frappe.model.document import Document
import base64
import requests
from openai import OpenAI
from google.cloud import vision
from google.oauth2 import service_account

class AutoInscribeIntegration(Document):
	'''Encapsulates a set of methods used to make external calls to OpenAI API & Google Vision API'''

	def get_openai_gpt_key(self):
		'''Returns decrypted OpenAI API key from AutoInscribe Settings'''
		
		return frappe.get_single("AutoInscribe Settings").get_password('openai_gpt_key')

	def get_vision_client_email(self):
		'''Returns Vision Client Email from AutoInscribe Settings'''
		
		return frappe.get_single("AutoInscribe Settings").as_dict()["vision_client_email"]

	def get_vision_project_id(self):
		'''Returns Vision Project ID from AutoInscribe Settings'''
		
		return frappe.get_single("AutoInscribe Settings").as_dict()["vision_project_id"]

	def get_vision_token_uri(self):
		'''Returns Vision Token URI from AutoInscribe Settings'''
		
		return frappe.get_single("AutoInscribe Settings").as_dict()["vision_token_uri"]
	
	def get_vision_private_key(self):
		'''Returns Vision Private Key from AutoInscribe Settings'''
		
		return frappe.get_single("AutoInscribe Settings").as_dict()["vision_private_key"]
	
	def ask_gpt(self, prompt):
		'''Returns response from OpenAI API given a prompt'''
		
		try:
			gpt_client = OpenAI(api_key=self.get_openai_gpt_key())
			chat_completion = gpt_client.chat.completions.create(
				messages=[
					{
						"role": "user",
						"content": prompt,
					}
				],
				model="gpt-3.5-turbo-1106",
			)
			return chat_completion.choices[0].message.content.strip()
		except Exception as e:
			frappe.throw("Please enter a valid OpenAI API key in AutoInscribe Settings")

	def extract_text_from_img(self, img_url):
		'''Extracts and returns first_name, middle_name, last_name, gender, salutation, designation contact_numbers, email_ids, company_name, website, address, mobile_number, phone_number, city, state and country from an image given the image URL'''
		
		try:
			credentials = service_account.Credentials.from_service_account_info({
				"type": "service_account",
				"project_id": self.get_vision_project_id(),
				"private_key": self.get_vision_private_key().strip().replace('\\n', '\n'),
				"client_email": self.get_vision_client_email(),
				"token_uri": self.get_vision_token_uri(),
			})
			response = requests.get(img_url)
			# Encode the image content to base64
			base64_img = base64.b64encode(response.content).decode('utf-8')
			client = vision.ImageAnnotatorClient(credentials=credentials)
			img_data = base64.b64decode(base64_img)
			# Create an image object
			image = vision.Image(content=img_data)
			# Perform OCR on the image
			response = client.text_detection(image=image)
			texts = response.text_annotations
		except Exception as e:
			frappe.throw("Please check your AutoInscribe Settings and try again")
		# Extracting detected text
		if texts:
			detected_text = texts[0].description
			prompt = f"From the following text, identify the first_name, middle_name, last_name, gender, salutation, designation contact_numbers, email_ids, company_name, website, address, mobile_number, phone_number, city, state, country: {detected_text}. Output must be a string containing one key-value pair per line and for absence of values use 'NULL' for value as placeholder. contact_numbers and email_ids must be comma-separated if there are multiple. Guess the salutation and gender. gender can be Male, Female, Transgender or Other. phone_number must be the telephone number whereas mobile_number must be the mobile number. country must have the value as full country name, e.g, US becomes United States, UK becomes United Kingdom."
			reply = self.ask_gpt(prompt)
			return reply
		else:
			return "No text detected"
	
	def create_address(self, address):
		'''Given an address string, extract city, state, postal_code, country and create an address if country exists & return the inserted doc. Return None otherwise.'''
		
		prompt = f"From the following address text, identify city, state, country and postal_code: {address}. Output must be a string containing one key-value pair per line and for absence of values use 'NULL'. country must have the value as full country name, e.g, US becomes United States, UK becomes United Kingdom"
		reply = self.ask_gpt(prompt)
		addr_lines = reply.strip().splitlines()
		city = addr_lines[0].split(':')[1].strip()
		state = addr_lines[1].split(':')[1].strip()
		postal_code = addr_lines[3].split(':')[1].strip()
		country = addr_lines[2].split(':')[1].strip()
		country_exists = frappe.db.exists("Country", {"country_name": country})
		if country_exists:
			doc = frappe.get_doc({
					"doctype": "Address",
					"address_title": address,
					"address_type": "Office",
					"address_line1": address,
					"city": city if city != "NULL" else None,
					"state": state if state != "NULL" else None,
					"country": country,
					"pincode": postal_code if postal_code != "NULL" else None,
					"is_your_company_address": 0
			})
			doc.insert()
			return doc
		else:
			return None

@frappe.whitelist()
def extract_text_from_img(img_url):
	doc = frappe.get_single("AutoInscribe Integration")
	return doc.extract_text_from_img(img_url)


@frappe.whitelist()
def create_address(address):
	doc = frappe.get_single("AutoInscribe Integration")
	return doc.create_address(address)

