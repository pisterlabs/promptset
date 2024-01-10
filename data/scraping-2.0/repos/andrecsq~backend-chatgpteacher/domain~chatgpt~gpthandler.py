import openai
from dotenv.main import dotenv_values

from domain.chatgpt.constants \
    import Models, Roles, API_KEY_PATH
from domain.chatgpt.prompts import generate_correction_prompt, generate_formatting_prompt

class GPTHandler:
    def __init__(self) -> None:
        envs = dotenv_values('.env')
        self.model = envs['DEFAULT_MODEL']
        openai.api_key_path = API_KEY_PATH

    def set_model_to_gpt3(self) -> None:
        self.model = Models.GPT3.value

    def set_model_to_gpt4(self) -> None:
        self.model = Models.GPT4.value

    def get_response_from_chat(self, prompt: list) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=prompt
        )

        content = response['choices'][0]['message']['content']
        return content.strip()
    
    def correct_translation(self, sentence_to_translate, translation_attempt):

        correction_prompt = generate_correction_prompt(sentence_to_translate, translation_attempt)
        
        correction_content = self.get_response_from_chat(correction_prompt)

        print("correction_content:")
        print(correction_content)

        formatting_prompt = generate_formatting_prompt(correction_prompt, correction_content)
        
        formatting_content = self.get_response_from_chat(formatting_prompt)

        print("formatting_content:")
        print(formatting_content)
        
        return formatting_content
