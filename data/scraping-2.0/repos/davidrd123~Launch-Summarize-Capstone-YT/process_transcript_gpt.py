import os
import openai
import tiktoken
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from pathlib import Path
import inquirer
import concurrent.futures

# System message for rewriting
SUMMARIZE_SYSTEM_MESSAGE = "[Assistant] I'm ready to help you analyze the YouTube video about the coding Capstone project. Please provide the transcript, and I'll produce a detailed explanation of all the parts of the system they are presenting, in the form of a well-structured markdown document, concluding with an ELI5 and then list out all the specific tools they used [Cloud Services] [Languages] [Other Technologies]"

REWRITE_SYSTEM_MESSAGE = "[Assistant] I understand the task at hand: to translate the given transcript of a coding Capstone project video from spoken to written language. I'll maintain all key technical details and try to preserve all the underlying information content. The goal is to create a written document that is about half the length of the original transcript but retaining all technical details. I will use up to 100 paragraphs to rewrite the transcript."

REWRITE_USER_MESSAGE = "This is a transcript of a coding Capstone project video. Please convert the oral communication into a well-written form that maintains as much detail as possible while reducing the length by a half, staying true to the essence of the content. I can use up to 100 paragraphs. The transcript is: "


def clean_transcript(transcript):
	# remove 'uh' and 'um' from transcript
	cleaned_transcript = transcript.replace('uh', '').replace('um', '')
	return cleaned_transcript

def process_transcript_in_chunks_parallel(transcript, chunk_size, overlap, model="gpt-3.5-turbo-16k", rewrite=False):
	# Calculate the total number of tokens
	total_tokens = len(transcript.split())

	# Initialize the start and end indices
	start = 0
	end = chunk_size

	# Initialize an empty list to hold the chunks of text
	chunks = []

	while start < total_tokens:
		# Get the chunk of text
		chunk = " ".join(transcript.split()[start:end])

		# Add the chunk to the list
		chunks.append(chunk)

		# Update the start and end indices for the next chunk
		start = end - overlap
		end = start + chunk_size
	
	# Initialize an empty list to hold the processed texts
	processed_texts = []

	# Initialize a ThreadPoolExecutor
	with concurrent.futures.ThreadPoolExecutor() as executor:
		# Use the executor to map the get_completion function to the chunks
		for processed_text in executor.map(lambda chunk: get_completion(text=chunk, model=model, rewrite=rewrite), chunks):
			print(f"Received completion: {processed_text[:50]}")
			# Add the processed text to the list
			processed_texts.append(processed_text)

	# Combine the processed texts into a single string
	final_text = " ".join(processed_texts)

	return final_text



def process_transcript_in_chunks(transcript, chunk_size, overlap, model="gpt-3.5-turbo-16k", rewrite=False):
	# Initialize the list of processed texts
	processed_texts = []

	# Calculate the total number of tokens
	total_tokens = len(transcript.split())

	# Initialize the start and end indices
	start = 0
	end = chunk_size

	while start < total_tokens:
		# Get the chunk of text
		chunk = " ".join(transcript.split()[start:end])
        
        # Generate a processed text for this chunk
		processed_text = get_completion(text=chunk, model=model, rewrite=rewrite)
        
		print(f"Processing chunk {chunk[:100]} to {processed_text[:100]}")

        # Add the processed text to the list
		processed_texts.append(processed_text)
        
        # Update the start and end indices for the next chunk
		start = end - overlap
		end = start + chunk_size
    
	# Combine the processed texts into a single string
	final_text = " ".join(processed_texts)

	return final_text

def get_completion(text, model="gpt-3.5-turbo-16k", rewrite=False):
	print(f"Getting completion for: (rewrite = {rewrite})| {text[:100]}")

	user_message = REWRITE_USER_MESSAGE + text if rewrite else f"This is a transcript of a coding Capstone project video. Please analyze it and provide a detailed explanation of the capstone project as a well-structured markdown document, including an ELI5 summary, and a list of the specific tools used. The transcript is: {text}"
	system_message = REWRITE_SYSTEM_MESSAGE if rewrite else SUMMARIZE_SYSTEM_MESSAGE

	user_message_token_count = get_token_count(user_message)
	system_message_token_count = get_token_count(system_message) 

	token_count = user_message_token_count + system_message_token_count

	print(f"Token count: {token_count}")

	max_tokens = 16000 - token_count - 50 if model == "gpt-3.5-turbo-16k" else 8191 - token_count - 50

	print(f"Max tokens: {max_tokens}")

	messages = [
		{"role": "user", "content": user_message},
		{"role": "system", "content": system_message}
	]
	response = openai.ChatCompletion.create(
		model=model,
		messages=messages,
		temperature=1,
		max_tokens=max_tokens,
	)
	print(f"Got response: {response}")
	return response.choices[0].message["content"]

def get_token_count(transcript):
	encoding = tiktoken.get_encoding('cl100k_base')

	# Count the tokens in the transcript
	token_count = len(encoding.encode(transcript))
	print(f"Token count: {token_count}")

	return token_count


def select_model():
	models = ['gpt-3.5-turbo-16k', 'gpt-4-0613']

	questions = [
		inquirer.List('chosen_model',
					message="Select a model",
					choices=models,
					)
	]
	
	answers = inquirer.prompt(questions)
	return answers['chosen_model']


def select_option():
	options = [
		'Rewrite Transcript in Shorter Form', 
		'Summarize Rewrite in Outline Form',
		'Summarize Transcript in Outline Form', 
		'Get token count of Transcript',
		'Get token count of Rewrite',
	]

	questions = [
		inquirer.List('chosen_option',
					message="Select an option",
					choices=options,
					)
	]
	
	answers = inquirer.prompt(questions)
	return answers['chosen_option']

def select_directory(subdirs):

	# Create a list of subdir names
	subdir_names = [subdir.name for subdir in subdirs]

	# Create a list prompt
	questions = [
		inquirer.List('chosen_dir',
						message="Select a directory",
						choices=subdir_names,
						)
	]

	# Get the user's answer
	answers = inquirer.prompt(questions)

	# Find the selected subdir
	selected_dir = next(subdir for subdir in subdirs if subdir.name == answers['chosen_dir'])
	return selected_dir



def app():
	transcripts_path = Path('transcripts')

	# Get a list of all (year) subdirectories in the transcripts folder
	year_dirs = sorted([d for d in transcripts_path.iterdir() if d.is_dir()])

	# Ask the user to select a year
	selected_year_dir = select_directory(year_dirs)

	# Get a list of all subdirectories within the selected year directory, sorted by name
	video_dirs = sorted([d for d in selected_year_dir.iterdir() if d.is_dir()], key=lambda d: d.name)

	# Ask the user to select a directory
	selected_video_dir = select_directory(video_dirs)

	# Should only be one transcript ending in _transcript.txt
	transcript_path = [f for f in selected_video_dir.iterdir() if f.name.endswith('_transcript.txt')][0]

	# Read the transcript
	with transcript_path.open('r', encoding='utf-8') as f:
		transcript = f.read()

	# Clean the transcript
	cleaned_transcript = clean_transcript(transcript)

	chosen_model = select_model()
	chosen_option = select_option()

	if chosen_option == 'Summarize Transcript in Outline Form':
		processed_text = get_completion(cleaned_transcript, chosen_model)
	elif chosen_option == 'Rewrite Transcript in Shorter Form':
		# processed_text = get_completion(cleaned_transcript, chosen_model, rewrite=True)
		processed_text = process_transcript_in_chunks_parallel(cleaned_transcript, chunk_size=2500, overlap=500, model=chosen_model, rewrite=True)
	elif chosen_option == 'Summarize Rewrite in Outline Form':
		# Check if there is a rewrite file in the directory
		rewrite_path = sorted([f for f in selected_video_dir.iterdir() if f.name.startswith('rewrite_')])[-1] # Get the most recent rewrite
		if rewrite_path:
			# Read the rewrite
			with rewrite_path.open('r', encoding='utf-8') as f:
				rewrite = f.read()
			processed_text = get_completion(rewrite, chosen_model)
		else:
			# Rewrite the transcript
			processed_text = get_completion(cleaned_transcript, chosen_model, rewrite=True)
			# Save the rewrite to a file in the same directory as the transcript with name 'rewrite_YYYYMMDDHHMMSS.txt'
			current_time = datetime.now().strftime("%Y%m%d%H%M%S")
			save_path = transcript_path.parent / f'rewrite_{current_time}.txt'
			with save_path.open('w', encoding='utf-8') as f:
				f.write(processed_text)
			
			# Then summarize the rewrite
			processed_text = get_completion(processed_text, chosen_model)
	elif chosen_option == 'Get token count of Transcript':
		token_count = get_token_count(cleaned_transcript)
		# Write the token count to a file
		with open(selected_video_dir / 'num_transcript_tokens.txt', 'w') as f:
			f.write(str(token_count))
		return # Exit the function, as we're not summarizing or rewriting
	elif chosen_option == 'Get token count of Rewrite':
		# Check if there is a rewrite file in the directory
		rewrite_path = sorted([f for f in selected_video_dir.iterdir() if f.name.startswith('rewrite')])[-1]
		if rewrite_path:
			# Read the rewrite
			with rewrite_path.open('r', encoding='utf-8') as f:
				rewrite = f.read()
			token_count = get_token_count(rewrite)
			print(f'Token count of rewrite {rewrite_path}: {token_count}')
			return # Exit the function, as we're not summarizing or rewriting
		else:
			print('No rewrite file found, processing transcript first')
			# Rewrite the transcript
			processed_text = get_completion(cleaned_transcript, chosen_model, rewrite=True)
			# Save the rewrite to a file in the same directory as the transcript with name 'rewrite_YYYYMMDDHHMMSS.txt'
			current_time = datetime.now().strftime("%Y%m%d%H%M%S")
			save_path = transcript_path.parent / f'rewrite_{current_time}.txt'
			with save_path.open('w', encoding='utf-8') as f:
				f.write(processed_text)
			token_count = get_token_count(processed_text)
			print(f'Token count of rewrite: {token_count}')
			# Write the token count to a file
			with open(selected_video_dir / 'num_rewrite_tokens.txt', 'w') as f:
				f.write(str(token_count))
			return # Exit the function, as we're not summarizing or rewriting



	current_time = datetime.now().strftime("%Y%m%d%H%M%S")
	if chosen_option == 'Summarize Transcript in Outline Form':
		# Save the summary to a file in the same directory as the transcript with name 'summary_YYYYMMDDHHMMSS.md'
		save_path = transcript_path.parent / f'summary_transcript_{current_time}.md'
	elif chosen_option == 'Rewrite Transcript in Shorter Form':
		# Save the summary to a file in the same directory as the transcript with name 'rewrite_YYYYMMDDHHMMSS.txt'
		save_path = transcript_path.parent / f'rewrite_{chosen_model}_{current_time}.txt'
	else: # Summarize Rewrite in Outline Form
		# Save the summary to a file in the same directory as the transcript with name 'summary_YYYYMMDDHHMMSS.md'
		save_path = transcript_path.parent / f'summary_rewrite_{chosen_model}_{current_time}.md'
	
	print(save_path)

	print(processed_text)

	with save_path.open('w', encoding='utf-8') as f:
		f.write(processed_text)
	


if __name__ == '__main__':	
	app()