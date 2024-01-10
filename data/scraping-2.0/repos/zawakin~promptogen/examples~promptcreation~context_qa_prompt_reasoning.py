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
# ContextualQuestionAnswering: (context: str, question: str) -> (answer: str)

# Description
# -----------
# Answer the question for the given context.

# Input Parameters
# ----------------
# - context (str): The contextual information
# - question (str): The question based on the context

# Output Parameters
# -----------------
# - answer (str): The answer to the question

# Examples Count
# --------------
# 1

from promptogen.prompt_tool import PromptWithReasoningTransformer

reasoning_extractor = TextLLMReasoningExtractor(
    text_llm=llm,
    reasoning_template="This is because ... So the answer is ...",
)
transformer = PromptWithReasoningTransformer(reasoning_extractor=reasoning_extractor)

context_qa_prompt_with_reasoning = transformer.transform_prompt(context_qa_prompt)
print(context_qa_prompt_with_reasoning)
# ContextualQuestionAnswering: (context: str, question: str) -> (reasoning: str, answer: str)

# Description
# -----------
# Answer the question for the given context.

# Input Parameters
# ----------------
# - context (str): The contextual information
# - question (str): The question based on the context

# Output Parameters
# -----------------
# - reasoning (str): Reasoning for the output
# - answer (str): The answer to the question

# Examples Count
# --------------
# 1

input_value = {
    "context": "The quick brown fox jumps over the lazy dog.",
    "question": "What does the fox jump over?",
}

output_value = prompt_runner.run_prompt(context_qa_prompt_with_reasoning, input_value=input_value)
print(output_value)
# {'reasoning': 'This is because the context provides information that the quick brown fox jumps over the lazy dog. The question asks what the fox jumps over. Therefore, the answer is that the fox jumps over the lazy dog.', 'answer': 'The fox jumps over the lazy dog.'}
