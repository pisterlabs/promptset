from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import List


def load_model() -> LlamaCpp:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers = 20
    n_batch = 512
    llm: LlamaCpp = LlamaCpp(
        model_path="resources/llama-2-7b-chat.gguf.q5_K_S.bin",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        f16_kv=True,
        temperature=0.7,
        max_tokens=1000,
        top_p=0.7,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    return llm


def create_prompt(template: str, input_var: List[str]) -> PromptTemplate:
    return PromptTemplate(template=template, input_variables=input_var)


def create_few_shot_prompt(classification_examples: List[dict], prompt: PromptTemplate, input_var: List[str],
                           prefix: str, suffix: str, sep: str) -> FewShotPromptTemplate:
    return FewShotPromptTemplate(examples=classification_examples, example_prompt=prompt, prefix=prefix, suffix=suffix,
                                 input_variables=input_var, example_separator=sep)


def classify_attribute(attribute_name: str, attribute_instances: str, few_shot_prompt: FewShotPromptTemplate,
                       model: LlamaCpp):
    result = model(few_shot_prompt.format(name=attribute_name, elements=attribute_instances))
    return result.split("\n")[0]


def get_func_variables_from_pandas(df):
    my_dict = {}
    for col in df.columns:
        my_dict[col] = df[col].values[:10].tolist()
    return my_dict
