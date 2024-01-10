import promptogen as pg
from examples.llm.openai_util import OpenAITextLLM
from promptogen.prompt_collection import PromptCreatorPrompt

llm = OpenAITextLLM(model="gpt-3.5-turbo")

formatter = pg.KeyValuePromptFormatter()
prompt_runner = pg.TextLLMPromptRunner(llm=llm, formatter=formatter)

prompt_creator_prompt = PromptCreatorPrompt()


def setup_language_translator(source_language: str = "English", target_language: str = "Japanese") -> pg.Prompt:
    input_value = {
        "description": f"Translate the given text from {source_language} to {target_language}.",
        "background": "(text: str) -> (translated_text: str)",
    }
    resp = prompt_runner.run_prompt(prompt_creator_prompt, input_value=input_value)
    return pg.Prompt.from_dict(resp["prompt"])


language_translator_prompt = setup_language_translator()
print(language_translator_prompt.to_dict())

input_value = {
    "text": "The quick brown fox jumps over the lazy dog.",
}

output_value = pg.TextLLMPromptRunner(llm=llm, formatter=formatter).run_prompt(
    language_translator_prompt, input_value=input_value
)

print(output_value["translated_text"])
# -> "速い茶色のキツネはのんびりした犬を飛び越えます。"
