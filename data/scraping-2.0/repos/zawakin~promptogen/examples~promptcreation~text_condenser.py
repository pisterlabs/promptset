import promptogen as pg
from examples.llm.openai_util import OpenAITextLLM
from promptogen.prompt_collection import PromptCreatorPrompt
from promptogen.prompt_tool import TextLLMReasoningExtractor

llm = OpenAITextLLM(model="gpt-3.5-turbo")
smart_llm = OpenAITextLLM(model="gpt-4")

formatter = pg.KeyValuePromptFormatter()
prompt_runner = pg.TextLLMPromptRunner(llm=llm, formatter=formatter)
smart_prompt_runner = pg.TextLLMPromptRunner(llm=smart_llm, formatter=formatter)

prompt_creator_prompt = PromptCreatorPrompt()


def setup_text_condenser_prompt() -> pg.Prompt:
    input_value = {
        "description": "Given a long text and a piece of information, condense the text into a shorter version while ensuring the specified information is preserved. Provide the condensed version of the text.",
        "background": "(text: str, information: str) -> (shorter_text: str); add at least three examples",
    }
    resp = smart_prompt_runner.run_prompt(prompt_creator_prompt, input_value=input_value)
    return pg.Prompt.from_dict(resp["prompt"])


text_condenser_prompt = setup_text_condenser_prompt()
print(text_condenser_prompt.to_dict())

input_value = {
    "text": """Large typhoon No. 6 is expected to approach the Okinawa/Amami area from August 31 to August 1 with strong force. Okinawa will be in the storm zone with wind speeds of 25 meters or higher on the 31st, and Amami will have very strong winds. The sea is expected to be very cold.
The maximum instantaneous wind speed on the 31st is estimated to be 35 meters for both Okinawa and Amami, and 40-60 meters for Okinawa and 35-45 meters for Amami on the 1st. There is a possibility that trucks may overturn during the storm, so please refrain from going out unnecessarily and be on high alert for storms and high waves from the 31st to the 1st. Also, very heavy rain is expected to fall in some areas, and heavy rains are likely. Be on the lookout for landslides and other disasters.""",
    "information": "wind speed",
}

output_value = prompt_runner.run_prompt(text_condenser_prompt, input_value=input_value)

print(output_value["shorter_text"])
# # -> The fox jumps over the lazy dog.

# Generate reasoning for the answer
reasoning_extractor = TextLLMReasoningExtractor(
    text_llm=llm,
    reasoning_template="This is because ... So the answer is ...",
)

print(
    reasoning_extractor.generate_reasoning(
        prompt=text_condenser_prompt,
        example=pg.IOExample(
            input=input_value,
            output=output_value,
        ),
    ).reasoning
)
# -> This is because the input text "context" provides the information that the quick brown fox jumps over the lazy dog. The input question "question" asks what the fox jumps over. Therefore, the answer is "The fox jumps over the lazy dog" which is derived directly from the context information.
