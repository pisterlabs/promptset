import promptogen as pg
from examples.llm.openai_util import OpenAITextLLM
from promptogen.prompt_collection import PromptCreatorPrompt
from promptogen.prompt_tool import TextLLMReasoningExtractor

llm = OpenAITextLLM(model="gpt-3.5-turbo")

formatter = pg.KeyValuePromptFormatter()
prompt_runner = pg.TextLLMPromptRunner(llm=llm, formatter=formatter)

prompt_creator_prompt = PromptCreatorPrompt()


def setup_context_qa_prompt() -> pg.Prompt:
    input_value = {
        "description": "Answer the question for the given context.",
        "background": "(context: str, question: str) -> (answer: str)",
    }
    resp = prompt_runner.run_prompt(prompt_creator_prompt, input_value=input_value)
    return pg.Prompt.from_dict(resp["prompt"])


context_qa_prompt = setup_context_qa_prompt()
print(context_qa_prompt)

input_value = {
    "context": "The quick brown fox jumps over the lazy dog.",
    "question": "What does the fox jump over?",
}

output_value = prompt_runner.run_prompt(context_qa_prompt, input_value=input_value)

print(output_value["answer"])
# -> The fox jumps over the lazy dog.

# Generate reasoning for the answer. This is done by using the LLMReasoningExtractor.
reasoning_extractor = TextLLMReasoningExtractor(
    text_llm=llm,
    reasoning_template="This is because ... So the answer is ...",
)

print(
    reasoning_extractor.generate_reasoning(
        prompt=context_qa_prompt,
        example=pg.IOExample(
            input=input_value,
            output=output_value,
        ),
    ).reasoning
)
# -> This is because the input text "context" provides the information that the quick brown fox jumps over the lazy dog. The input question "question" asks what the fox jumps over. Therefore, the answer is "The fox jumps over the lazy dog" which is derived directly from the context information.
