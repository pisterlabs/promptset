from lib.OpenAIRequester import OpenAIRequester

class ChooseBestSummary(OpenAIRequester):
    def __init__(self, model, output_file, sentences = 5, words = 80):
        super().__init__('ChooseBestSummary', model, output_file)
        self.sentences = sentences
        self.words = words

    def get_system_message(self):
        return {
           "role": "system",
           "content": f"""I will provide you with piece of content (e.g. articles, papers, documentation, etc.)

           Step 1.
           You will generate 5 unique concise, entity-dense summaries of the content.

           Step 2.
           Choose the summary which best matches the original content by considering all prior summaries.

           Guidelines:

           All summaries should be long ({self.sentences} sentences, -{self.words} words).
           Each summary should provide a different perspective on the content.

           Answer in JSON. The JSON should be a list (length 6 - the first five summaries and the best summary) of dictionaries whose key is "Summary"."""
       }


    def get_user_message(self, input_text, warn=False):
        warning = ""
        if warn is True:
            warning = "In the last response, you have missed that the response should be a JSON array.  Please make sure that the response is a JSON object with an array of objects each containing a 'Summary' key as described in the system prompt.  "

        return {
                "role": "user",
                "content": f"{warning}Here is the input text for you to summarise using the 'Summary' approach:\n\n{input_text}",
            }

    def parse(self, input_text):
        arr = self.parse_into_array(input_text, 'Summary')
        if arr is not None:
            return arr[-1]
        return None
