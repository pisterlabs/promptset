import json
from typing import Dict, Any, List, Iterable

from langchain.chains import LLMChain
from langchain.chat_models import ChatVertexAI
from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser, RegexParser
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.schema import AIMessage, BaseOutputParser, PromptValue
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

try:
    import config
except ModuleNotFoundError:
    import finetune.config as config


class RatableText(BaseModel):
    context_text: str
    output: str


class RatingOutput(BaseModel):
    rating: float = Field(description="A rating between 1 and 5", ge=1, le=5)
    thought: str = Field(description="Thought process for the rating")


class ChromaRatingExampleSelector(BaseExampleSelector):

    def __init__(self, collection, low_k=1, high_k=1):
        BaseExampleSelector.__init__(self)
        self.collection = collection
        self.num_low = low_k
        self.num_high = high_k

    def add_example(self, example: Dict[str, str]) -> Any:
        raise NotImplementedError("Read only")

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        high_result = self.collection.query(query_texts=[input_variables["text"]], where={"classified": "h"},
                                            n_results=self.num_high)
        low_result = self.collection.query(query_texts=[input_variables["text"]], where={"classified": "l"},
                                           n_results=self.num_low)
        return self._convert_query_result(high_result) + self._convert_query_result(low_result)

    def _convert_query_result(self, result) -> List[dict]:
        results = []
        for text, metadata in zip(result["documents"][0], result["metadatas"][0]):
            results.append({
                "context_text": text,
                "output": metadata["output"],
                "rating": metadata["rating"],
                "thought": metadata["thought"]
            })
        return results


class RandomExampleSelector(BaseExampleSelector):

    def __init__(self, examples: List[Dict[str, str]], k=3):
        self.examples = examples
        self.k = k

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        return np.random.choice(self.examples, size=self.k, replace=False)


class PromptAdoptingParser(BaseOutputParser):
    base_parser: BaseOutputParser
    prompt: PromptValue

    def parse_with_prompt(self, completion: str, prompt_value: PromptValue):
        return self.base_parser.parse_with_prompt(completion, prompt_value)

    def parse(self, completion: str):
        return self.parse_with_prompt(completion, self.prompt)

    def get_format_instructions(self):
        return self.base_parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return self.base_parser.type


def create_label_examples(examples: Iterable[Dict[str, str]], user_prompt, reinforcement=config.LABEL_REINFORCEMENT):
    result = []
    human_template = HumanMessagePromptTemplate.from_template(reinforcement + user_prompt)
    ai_format = "Thought process:\n{thought}\n\nFinal Rating:\n{rating}"
    for e in examples:
        result.append(human_template.format(context=e["context_text"], output=e["output"]))
        result.append(AIMessage(content=ai_format.format(thought=e["thought"], rating=e["rating"])))
    return result


def rate_data_raw(model: BaseChatModel, text: RatableText, system: str, user: str,
                  example_selector: BaseExampleSelector = None):
    examples = []
    if example_selector:
        selected = example_selector.select_examples({"text": text.context_text})
        examples = create_label_examples(selected[1:],
                                         user_prompt=user)
        examples = create_label_examples(selected[:1], user_prompt=user, reinforcement="") + examples
    system_template = SystemMessagePromptTemplate.from_template(system)
    if len(examples) > 0:
        user = config.LABEL_REINFORCEMENT + user
    human_template = HumanMessagePromptTemplate.from_template(user)
    prompt = ChatPromptTemplate.from_messages([system_template, *examples,
                                               human_template])

    prompt_val = prompt.format_prompt(context=text.context_text, output=text.output)
    token_counts = model.get_num_tokens_from_messages(prompt_val.to_messages())
    if token_counts > config.LABEL_MAX_TOKEN:
        raise ValueError("Token count {count} > {max_token}".format(count=token_counts,
                                                                    max_token=config.LABEL_MAX_TOKEN))

    raw_chain = LLMChain(llm=model, prompt=prompt, verbose=config.LABEL_VERBOSE)
    raw_text = raw_chain(inputs={"context": text.context_text, "output": text.output})["text"]
    return raw_text


def create_format_examples(examples: Iterable[Dict[str, str]]):
    result = []
    human_prompt = HumanMessagePromptTemplate.from_template(config.LABEL_FORMAT_USER)

    for e in examples:
        valid_obj = RatingOutput(rating=e["rating"], thought=e["thought"])
        result.append(human_prompt.format(raw=e["raw"]))
        result.append(AIMessage(content=json.dumps(valid_obj.model_dump())))

    return result


def format_appropriate_meal(model: BaseChatModel, raw_text: str, examples: Iterable[Dict[str, str]] = ()):
    format_prompt = ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate.from_template(config.LABEL_FORMAT_SYSTEM),
         *create_format_examples(examples),
         HumanMessagePromptTemplate.from_template(config.LABEL_FORMAT_USER)])
    output_parser = RetryWithErrorOutputParser.from_llm(
        parser=PydanticOutputParser(pydantic_object=RatingOutput),
        llm=model)
    try:
        format_prompt = format_prompt.partial(format_instructions=output_parser.get_format_instructions())
    except NotImplementedError:
        format_prompt = format_prompt.partial(format_instructions="")
    prompt_val = format_prompt.format_prompt(raw=raw_text)
    token_counts = model.get_num_tokens_from_messages(prompt_val.to_messages())
    if token_counts > config.LABEL_MAX_TOKEN:
        raise ValueError("Token count {count} > {max_token}".format(count=token_counts,
                                                                    max_token=config.LABEL_MAX_TOKEN))
    output_parser = PromptAdoptingParser(base_parser=output_parser, prompt=prompt_val)
    parse_chain = LLMChain(llm=model, prompt=format_prompt, output_parser=output_parser,
                           verbose=config.LABEL_VERBOSE)
    result = parse_chain(inputs={"raw": raw_text})["text"].model_dump()
    result["raw"] = raw_text
    return result


def evaluate_text(model: BaseChatModel, texts: RatableText, system: str, user: str,
                  format_selector: BaseExampleSelector = None,
                  meal_selector: BaseExampleSelector = None):
    format_examples = []
    if format_selector:
        format_examples = format_selector.select_examples(input_variables={})

    raw_text = rate_data_raw(model, texts, system, user, meal_selector)
    formatted = format_appropriate_meal(model, raw_text, format_examples)
    return formatted


def create_db_entries(records, col):
    cur_id = col.count()
    ids = []
    documents = []
    classified = []
    thought = []
    summary = []
    ratings = []
    for idx, r in records.iterrows():
        documents.append(r["body"])
        if r["rating"] > 3:
            classified.append("h")
        else:
            classified.append("l")
        thought.append(r["thought"])
        ratings.append(r["rating"])
        summary.append(r["predicted"])
        ids.append(str(cur_id))
        cur_id += 1

    return {
        "documents": documents,
        "ids": ids,
        "metadatas": [{"thought": t, "classified": c, "rating": v, "output": w} for t, c, v, w in
                      zip(thought, classified, ratings, summary)]
    }


def augment_docs(chain, docs, qrel):
    cur_doc = docs.doc_id.max() + 1
    augment_docs = docs.doc_text.progress_apply(lambda x: chain(x)["text"])
    doc_id_texts = augment_docs.to_frame().reset_index(drop=True)
    doc_id_texts["orig_id"] = docs.doc_id.tolist()
    doc_id_texts["new_id"] = cur_doc + doc_id_texts.index
    doc_qrel_temp = pd.merge(left=doc_id_texts, right=qrel, how="inner", left_on="orig_id", right_on="doc_id")
    aug_qrel = doc_qrel_temp[["query_id", "new_id", "relevance"]].rename(columns={"new_id": "doc_id"})
    aug_docs = doc_id_texts[["new_id", "doc_text"]].rename(columns={"new_id": "doc_id"})
    return aug_docs, aug_qrel
