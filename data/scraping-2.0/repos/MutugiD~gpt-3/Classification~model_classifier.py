import openai


def set_openai_key(key):
    """Sets OpenAI key."""
    openai.api_key = key
 
class Classifier:
    def __init__(self):
        print("Report Generator Model Intialized--->")

    def classification(self, input_text):
        """
        wrapper for the API to save the input text and the generated report
        """        
        # Arguments to send the API
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Determine the sentiment of the following review:\n{input_text}",
            temperature=0,
            max_tokens=64,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response["choices"][0]["text"].strip()

    def model_prediction(self, input_text, api_key):
        """
        wrapper for the API to save the prompt and the result
        """
        set_openai_key(api_key)
        sentiment = self.classification(input_text)   
        return sentiment 

