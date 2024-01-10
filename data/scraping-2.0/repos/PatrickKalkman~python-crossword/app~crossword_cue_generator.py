from app.openai_response_processor import OpenAIResponseProcessor


class CrosswordCueGenerator(OpenAIResponseProcessor):
    def _create_prompt(self, answer):
        return f"Generate a clue for the crossword answer '{answer}'."

    def _parse_response(self, response_text):
        return response_text.strip().replace("\"", "")

    def _create_cache_filename(self, answer):
        return f"cache/{answer}.json"
