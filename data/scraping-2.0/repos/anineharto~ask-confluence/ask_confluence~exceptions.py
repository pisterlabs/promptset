from openai.error import OpenAIError

class AnswerNotFoundError(OpenAIError):
    """Error for missing information among uploaded data to openAI."""
    pass