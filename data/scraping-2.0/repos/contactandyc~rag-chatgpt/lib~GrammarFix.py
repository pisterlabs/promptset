from lib.OpenAIRequester import OpenAIRequester

class GrammarFix(OpenAIRequester):
    def __init__(self, model, output_file):
        super().__init__('GrammarFix', model, output_file)

    def get_system_message(self):
        return {
            "role": "system",
            "content": f"""I will provide you with piece of content (e.g. articles, papers, documentation, etc.)

            You will regenerate the text so that proper punctuation, paragraph, and sentence breaks exist.  You will
            also fix any grammar issues and fill in any missing words.  Please remove words like "Um" and repeated
            text which is not grammatically correct.

            Guidelines:

            The regenerated text must be complete.

            Answer in JSON. The JSON should contain the regenerated text as with the key "Text"."""}

    def get_user_message(self, input_text, warn=False):
        warning = ""
        if warn is True:
            warning = "In prior responses, you have missed that the response should be a JSON object.  Please make sure that the response is a JSON object with a single key 'Text' as described in the system prompt.  "

        return {
                "role": "user",
                "content": f"{warning}Here is the input text for you to correct using the 'Text' approach:\n\n{input_text}",
            }

    def parse(self, input_text):
        return self.parse_value_from_object(input_text, 'Text')
