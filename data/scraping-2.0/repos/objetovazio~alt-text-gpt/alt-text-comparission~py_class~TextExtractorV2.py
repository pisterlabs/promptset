import os
import logging
import requests
import openai
from .TextExtractorV1 import TextExtractorV1

class TextExtractorV2(TextExtractorV1):
	def __init__(self, get_from_file):
		super().__init__(get_from_file)
		self.ep_url = os.getenv('EVERYPIXEL_API_URL')  # get the EveryPixel API URL from environment variables
		self.ep_username = os.getenv('EVERYPIXEL_USERNAME')  # get the EveryPixel username from environment variables
		self.ep_api_key = os.getenv('EVERYPIXEL_API_KEY')  # get the EveryPixel API key from environment variables
		self.open_api_key = os.getenv('OPEN_API_KEY')

		self.auth = (self.ep_username, self.ep_api_key)

	def _get_prompt(self, salesforce_caption, keywords, language="Portuguese"):
		prompt = f"""
Act as an image analyzer and advanced caption writer based on text params. 
Inputs, explanations, and values:

original_caption: Pre-generated image caption in English
value: {salesforce_caption}

image_tags: Image tags followed by their scores, in the format [('word1', score1)]. The tags can refer to anything in the image, such as people, objects, activities, colors, etc.
value: {keywords}

output_language: The language in which the output caption should be generated.
value: {language}

generated_caption:
After processing the inputs, return generated_caption in the formated as "Result: [generated_caption_placeholder]"

Follow the steps to process the inputs, as outlined above.
1. Read the original_caption and analyze the context. Read the image_tags and sort them by score from the highest to the lowest score. If tags are empty, proceed to step 4.

2. Iterate over each tag and analyze if it can be added to the context of the original_caption to improve the caption. When a tag is evaluated, integrate it into the original_caption using one of the following methods to generate a result based on the tag:
   - Image type: Indicate if it's a photograph, cartoon, comic strip, or illustration. Provide a brief description with up to 4 words.
   - Characteristics: If referring to a person, indicate nouns referring to age, appearance, size, etc. If referring to an object, add its information.
   - Ethnicity: Provide more detailed treatment for ethnicities, indicating one or two characteristics. Describe skin color using IBGE terms: white, black, brown, indigenous, or yellow.
   - Hair: Provide a description of the type and color using synonyms.
   - Clothing: Provide a description of the type and color using synonyms.
   - Object: Provide a brief description with up to 4 words about the object's characteristics.
   - Object in use: Provide a brief description with up to 4 words about the action involving the object.
   - Environment: Provide a brief description with up to 4 words about the environment.

3. Perform a double-check on the original_caption, generated_caption, and image_tags. Ensure that all information present in the generated_caption is somehow present in the original caption or the tags.

4. Translate the final caption to the language specified in the 'output_language' field, considering cultural and regional nuances when applicable.

5. Return the result.
"""
		return prompt

	def extract_text_from_image_path(self, image_path):
		salesforce_caption, keywords = super().extract_text_from_image_path(image_path) # Chama Versao1
		
		# Add new steps here
		keywords = self.get_keywords_from_image_path(image_path)
		if keywords is not None:
			language = 'English'
			prompt = self._get_prompt(salesforce_caption, keywords, language)
			generated_text = self._call_chat_gpt_api(prompt)
			logging.info(f"Generated text: {generated_text}; Prompt: ({salesforce_caption}, {keywords}, {language})")
			return generated_text, keywords
	# end - extract_text_from_image_path()
		
	def extract_text_from_image_url(self, image_url):
		salesforce_caption = super().extract_text_from_image_url(image_url)

		# Add new steps here
		keywords = self.get_keywords_from_image_url(image_url)
		if keywords is not None:
			language = 'English'
			prompt = self._get_prompt(salesforce_caption, keywords, language)
			generated_text = self._call_chat_gpt_api(prompt)
			logging.info(f"Generated text: {generated_text}; Prompt: ({salesforce_caption}, {keywords}, {language})")
			return generated_text
	# end - extract_text_from_image_url()
	
	def get_keywords_from_image_path(self, image_path):
		logging.debug(f"Getting keywords from {image_path}.")
		try:
			with open(image_path, "rb") as image:
				data = {'data': image}
				params = {'num_keywords': 10}
				
				response = requests.post(
					self.ep_url,
					files=data,
					params=params,
					auth=self.auth
				)
				
			if response.status_code == 200:
				keywords = response.json().get('keywords', [])
				keyword_scores = [(keyword['keyword'], keyword['score']) for keyword in keywords if keyword['score'] >= 0.6]
				logging.info(f"Keywords successfully extracted from {image_path}: {keyword_scores}.")
				return keyword_scores
			else:
				logging.warning(f"Failed to get keywords from {image_path}.")
				return None
				
		except Exception as e:
			logging.error(f"Failed to extract keywords from {image_path} due to {str(e)}")
			return None
	# end - get_keywords_from_image_path()

	def get_keywords_from_image_url(self, image_url):
		logging.debug(f"Getting keywords from {image_url}.")
		try:
			params = {'url': image_url, 'num_keywords': 10}
			
			response = requests.get(
				self.ep_url,
				params=params,
				auth=self.auth
			)
			
			if response.status_code == 200:
				keywords = response.json().get('keywords', [])
				filtered_keywords = [(keyword['keyword'], keyword['score']) for keyword in keywords if keyword['score'] >= 0.6]
				logging.info(f"Keywords successfully extracted from {image_url}: {filtered_keywords}.")
				return filtered_keywords
			else:
				logging.warning(f"Failed to get keywords from {image_url}.")
				return None
				
		except Exception as e:
			logging.error(f"Failed to extract keywords from {image_url} due to {str(e)}")
			return None
	# end - get_keywords_from_image_url()

	def _call_chat_gpt_api(self, prompt):
		try:
			openai.api_key = self.open_api_key

			response = openai.ChatCompletion.create(
				# model="gpt-4",
				model="gpt-3.5-turbo",
				messages=[
					{
      					"role": "system",
      					"content": "Read the full description and dont say nothing but what is asked as return. You will act as an image analyzer and advanced caption writer based on text params."
					},
					{
						"role": "user",
						"content": f"{prompt}"
					},
				],
				n=1,
				temperature=0.6,
				max_tokens=500,
				frequency_penalty=0,
				presence_penalty=0,
			)

			generated_text = response.choices[0].message.content
			generated_text = generated_text.split('Result: ')[1].strip("\"'\n ,.")
			return generated_text
		except Exception as e:
			logging.error(f"Failed to call ChatGPT API due to {str(e)}")
			raise Exception(f"Failed to call ChatGPT API due to {str(e)}")
	