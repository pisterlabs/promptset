import openai
import os
from dotenv import load_dotenv

def main():
	load_dotenv()
	print ("Starting Training...")
	script_directory = os.path.dirname(os.path.abspath(__file__))
	relative_training = "training_data.jsonl" 

	training_path = os.path.join(script_directory, relative_training)

	openai.api_key = os.getenv("OPENAI_API_KEY")
	training_response = openai.File.create(
	  file=open(training_path, "rb"),
	  purpose='fine-tune'
	)

	training_file_id = training_response["id"]

	response = openai.FineTuningJob.create(
			training_file=training_file_id,
			model="gpt-3.5-turbo",
			suffix="lwfm_assistant"
	)

	job_id = response["id"]

	print(response)

	response = openai.FineTuningJob.list_events(id=job_id, limit=50)

	events = response["data"]
	events.reverse()

	for event in events:
		print("EVENT MESSAGE: " + event["message"])

if __name__ == '__main__':
	main()
