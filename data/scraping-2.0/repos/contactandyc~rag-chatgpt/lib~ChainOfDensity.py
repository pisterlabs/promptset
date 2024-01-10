from lib.OpenAIRequester import OpenAIRequester

class ChainOfDensity(OpenAIRequester):
    def __init__(self, model, output_file, sentences = 5, words = 80):
        super().__init__('ChainOfDensity', model, output_file)
        self.sentences = sentences
        self.words = words

    def get_system_message(self):
        return {
           "role": "system",
           "content": f"""I will provide you with piece of content (e.g. articles, papers, documentation, etc.)

           You will generate increasingly concise, entity-dense summaries of the content.

           Repeat the following 2 steps 5 times.

           Step 1. Identify 1-3 informative Entities (";" delimited) from the Article which are missing from the previously generated summary.

           Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities.

           A Missing Entity is:

           Relevant: to the main story.
           Specific: descriptive yet concise (5 words or fewer).
           Novel: not in the previous summary.
           Faithful: present in the content piece.
           Anywhere: located anywhere in the Article.

           Guidelines:

           The first summary should be long ({self.sentences} sentences, -{self.words} words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach -{self.words} words.
           Make every word count: re-write the previous summary to improve flow and make space for additional entities.
           Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
           The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
           Missing entities can appear anywhere in the new summary.
           Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
           Remember, use the exact same number of words for each summary.
           Answer in JSON. The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary"."""
       }


    def get_user_message(self, input_text, warn=False):
        warning = ""
        if warn is True:
            warning = "In the last response, you have missed that the response should be a JSON array.  Please make sure that the response is a JSON object with an array of objects each containing a 'Missing_Entities' and a 'Denser_Summary' key as described in the system prompt.  "

        return {
                "role": "user",
                "content": f"{warning}Here is the input text for you to summarise using the 'Missing_Entities' and 'Denser_Summary' approach:\n\n{input_text}",
            }

    def parse(self, input_text):
        arr = self.parse_into_array(input_text, 'Denser_Summary')
        if arr is not None:
            return arr[-1]
        return None
