from pathlib import Path
import time
import openai
from ask_confluence.config import Config
import data.interim as interim_data_path
from ask_confluence.exceptions import AnswerNotFoundError

class ModelPredictor:
    """ModelPredictor class definition."""

    def __init__(self):
        """ModelPredictor init module."""
        self.__config = Config()
        self.interim_data_file_path = Path(interim_data_path.__file__).parent / "confluence_pages.jsonl"
    
    def _upload_file(self):
        """Upload file to OpenAI model.

        Returns:
            str: ID for uploaded file.
        """
        file_id = openai.File.create(file=open(self.interim_data_file_path), purpose="answers").get("id")
        time.sleep(7) # Issue with OpenAI returning file id before the processing is complete
        return file_id
    
    def _delete_file(self, file_id):
        """Delete file from OpenAI model.

        Args:
            file_id (str): ID of uploaded file to delete.
        """
        openai.File.delete(file_id)
    
    def get_answer(self, question):
        """Get answer to question using Open AI's search model.

        First, uploads file to be used among training data, retrieves
        answer to the given question, and finally deletes the file uploaded
        as training data (to limit the number of files uploaded).

        Args:
            question (str): Question.

        Returns:
            list(str): List of answers.
        """
        openai.api_key = self.__config.parameters["openai_api_key"]
        file_id = self._upload_file()
        try:
            answer = openai.Answer.create(
                search_model="ada",
                model="curie",
                question=question,
                file=file_id,
                examples_context=self.__config.parameters["examples_context"],
                examples=self.__config.parameters["examples"],
                max_tokens=5,
            ).get("answers")
        except openai.error.InvalidRequestError as error:
            raise AnswerNotFoundError("The answer could not be found among the uploaded documents.")
        
        self._delete_file(file_id=file_id)
        return answer
