import openai
import time
import os
from configuration.manage_secrets import ConfigurationManager

current_directory = os.path.dirname(os.path.abspath(__file__))
training_data_path = os.path.join(current_directory, 'gpt_training_data')

class OpenAIFineTuningJob:
    """
    Used to create a fine-tuning job on OpenAI's servers.
    """
    
    def __init__(self):
        self.api_key = ConfigurationManager().retrieve_api_keys()
        openai.api_key = self.api_key['OPENAI-API-KEY']
        
    def start_finetuning(self, model_name=None, file=None):
        """
        Begin the fine-tuning process.
        """
        # upload training data
        file_id = self._upload_training_data(file)
        
        # create fine-tuning job
        self._create_fine_tuning_job(file_id, model_name)

    def _upload_training_data(self, file) -> str:
        """
        Upload the training data to OpenAI's servers.
        """
        # Upload training data
        print(f"Uploading training data...")
        file_response = openai.File.create(
            file=file,
            purpose='fine-tune'
        )

        print(f"Training data uploaded successfully")
        print(f"File ID: {file_response['id']}")
        
        return file_response["id"]

    def _create_fine_tuning_job(self, file_id, model_name):
        """
        Begins a fine-tuning job.
        """
        if not model_name:
            model_name = input("Enter a name for the fine-tuned model: ") 
        
        print(f"Giving servers some time to process the file")
        time.sleep(20)
        
        print(f"Creating fine-tuning job...")
        while True:
            try:
                # Create a fine-tuning job
                fine_tuning_response = openai.FineTuningJob.create(
                    training_file=file_id,
                    model="gpt-3.5-turbo",
                    suffix=model_name
                )
                
                print(f"Fine-tuning job completed!")
                print(f"Job ID: {fine_tuning_response['id']}")

                return fine_tuning_response["id"]
            except openai.error.APIError as e:  
                print(f"Still waiting for servers to process the file")
                time.sleep(10)
            
if __name__ == "__main__":
    new_job = OpenAIFineTuningJob()
    new_job.start_finetuning()

