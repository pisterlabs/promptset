import promptogen as pg
from examples.llm.openai_util import OpenAITextLLM
from promptogen.prompt_collection.prompts.text_condenser import TextCondenserPrompt
from promptogen.prompt_tool import PromptWithReasoningTransformer, TextLLMReasoningExtractor

llm = OpenAITextLLM(model="gpt-3.5-turbo-16k")

formatter = pg.KeyValuePromptFormatter()
prompt_runner = pg.TextLLMPromptRunner(llm=llm, formatter=formatter)


text_condenser_prompt = TextCondenserPrompt()


def setup_reasoning_prompt(prompt: pg.Prompt) -> pg.Prompt:
    reasoning_extractor = TextLLMReasoningExtractor(
        text_llm=llm, reasoning_template="This is because ... So the answer is ..."
    )
    reasoning_transformer = PromptWithReasoningTransformer(reasoning_extractor)
    prompt_with_reasoning = reasoning_transformer.transform_prompt(prompt)
    return prompt_with_reasoning


text_condenser_prompt_with_reasoning = setup_reasoning_prompt(text_condenser_prompt)


def condense_text(text: str, information: str) -> str:
    input_value = {
        "information": information,
        "text": text,
    }
    resp = prompt_runner.run_prompt(text_condenser_prompt_with_reasoning, input_value=input_value)
    return resp["result"]


with open("examples/output/long_text.txt", "r") as f:
    text = f.read()

for i in range(10):
    print(condense_text(text, input("information: ")))
